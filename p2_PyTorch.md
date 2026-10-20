# CS336 P2 PyTorch

## 1. Scaling Law (缩放定律)

### 1.1 基本概念
**Scaling law (缩放定律)**: It's an empirical law in the LLM area. It describes how model performance changes as you increase resources such as:
* **Model size $N$**: number of parameters.
* **Training data $D$**: number of training tokens.
* **Training compute $C$**: training FLOPs.

A typical scaling law says that as you spend more compute, use a larger model, or train on more data, the loss tends to decrease in a fairly predictable way.

A simplified form is:
$$L(N, D) = L_{\infty} + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

* $L(N, D)$: model loss $\downarrow$
* $L_{\infty}$: the irreducible lower bound of loss
* $A, B, \alpha, \beta$: constants fitted (拟合) from experiments

### 1.2 缩放定律的作用与最优分配
Scaling law answers the question: *If I scale up, what happens to performance?*

For LLMs, this becomes:
$$\text{more parameters} + \text{more data} + \text{more compute} \xrightarrow{\text{key word: predictable}} \text{predictable reduction in loss}$$

Researchers can run many relatively small experiments, fit a scaling law, and then extrapolate to estimate what may happen at much larger training scales:

1. The researchers first train models at smaller compute budgets
2. $\rightarrow$ then estimate a relationship such as $L = f(N, D, C)$
3. $\rightarrow$ and use it to predict the loss of much larger models.

> **注意**：Scaling law does not mean "bigger is always better".

Suppose your total compute budget is fixed: $C \propto N \cdot D$.
There is usually an optimal allocation:
$$N_{\text{opt}}(C), \quad D_{\text{opt}}(C)$$
This is called **compute-optimal scaling**.

---

## 2. Resource Accounting (资源核算)

**Question:** What's the best model one can train given fixed resources (compute, memory)?
In other words: maximize (computational) efficiency.

### Question 1: 训练耗时估算
*How long would it take to train a $70\text{B}$ parameter model on $15\text{T}$ tokens on $1024$ H100s?*

```python
total_flops = 6 * 70e9 * 15e12
h100_flop_per_sec = 1979e12 / 2
mfu = 0.5  # Model FLOPs Utilization

flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24
days = total_flops / flops_per_day  # = 143.9266
```

* **旁注（关于 `total_flops` 的计算）**：
  $$C \approx 6ND = \underbrace{4ND}_{\text{backward pass}} + \underbrace{2ND}_{\text{forward pass}}$$

### Question 2: 显存与最大模型规模估算
*What's the largest model you can train on $8$ H100s using AdamW (naively)?*

```python
h100_bytes = 80e9
bytes_per_parameter = 2 + 2 + (4 + 4)  # parameters, gradient, optimizer state
num_parameters = (h100_bytes * 8) / bytes_per_parameter
```

* **旁注（关于 `2 + 2 + (4 + 4)` 的拆解）**：
  * 第一个 `2`：模型参数本身，例如使用 BF16 ($16\text{ bits} = 2\text{ bytes}$)。
  * 第二个 `2`：训练时，每个参数对应一个梯度，若梯度也为 BF16，则为 $2\text{ bytes}$。
  * `(4 + 4)`：代表 AdamW 的两个 optimizer states。AdamW 对每个参数维护两个重要变量：$m_t$（一阶矩）和 $v_t$（二阶矩），通常用 FP32 保存（各占 $4\text{ bytes}$）。

---

## 3. Tensors Basics (张量基础)

Tensors are the basic building block for storing everything.
张量包括向量、矩阵，甚至可以表示任意维度的数组（array）。

### 3.1 数组的维度与形状
避免形状不确定：

```python
a = [
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
]
```

* **维度解析**：
  * 最外层有 $2$ 个元素 $\rightarrow 2$
  * 每个元素里有 $2$ 个数组 $\rightarrow 2$
  * 每个数组里又有 $3$ 个数字 $\rightarrow 3$
  * $\text{shape} = (2, 2, 3)$
  * 由于需要 $3$ 个索引才能定位一个数字，故是 $3$ 维数组。例如：`a[0][0][1] = 2`。

### 3.2 浮点数表示 (`float32`)
`float32` 共 $32\text{ bits}$（从高到低为第 $31$ 位至第 $0$ 位）：
* **Bit $31$ (`sign`)**：符号位（$1\text{ bit}$）
* **Bits $30 \sim 23$ (`exponent`)**：指数位（$8\text{ bits}$），控制数字大致数量级
* **Bits $22 \sim 0$ (`fraction`)**：尾数/有效数字位（$23\text{ bits}$），控制精度

### 3.3 张量内存占用
Memory is determined by:
1. The number of values
2. Data type of each value

```python
x = torch.zeros(4, 8)
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8
assert x.element_size() == 4
assert get_memory_usage(x) == 4 * 8 * 4
```

* `float16`, `bfloat16`, `fp8`, `fp4` 这里就不一一介绍了，关键要根据**内存**、**数据范围**与**精度**合理选择。
* By default, tensors are stored in CPU memory. However, in order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory.

---

## 4. Computing with Tensors & `einops`

### 4.1 简介与动机
* **`einops` (Einstein operations)**: is a Python library designed for tensor rearrangement, reshaping, reduction, and repetition.
* **Motivation**: As tensors become higher-dimensional, operations such as `reshape`, `permute`, `transpose`, and `view` become difficult to read and easy to get wrong.
* **Core idea**: Describe tensor transformation using named dimensions, rather than relying only on numerical dimension indices.（灵感来源于爱因斯坦求和约定）

**Example**: Easy to mess up the dimensions (what's `-2`, `-1`?)...
Traditional PyTorch code:

```python
x = torch.ones(2, 2, 3)  # batch seq hidden
y = torch.ones(2, 2, 3)  # batch seq hidden
z = x @ y.transpose(-2, -1)  # batch seq seq
```

* **解析**：
  * `y.transpose(-2, -1).shape = (2, 3, 2)`
  * `x @ y.transpose(-2, -1)` 就是在两个 batch 中分别做矩阵乘法：$(2 \times 3) \text{ @ } (3 \times 2) \rightarrow 2 \times 2$
  * `z.shape = (2, 2, 2)`

### 4.2 `einops.einsum()`

```python
x = torch.ones(3, 4)  # seq1 hidden
y = torch.ones(4, 3)  # hidden seq2

# Old way:
z = x @ y  # seq1 seq2

# New way (einops):
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

* **解析**：`hidden` 类似于爱因斯坦求和约定中消去的对应矩阵维度。
  $$\text{eg: } x = \begin{pmatrix} \text{--- } x_1 \text{ ---} \\ \text{--- } x_2 \text{ ---} \\ \text{--- } x_3 \text{ ---} \end{pmatrix}, \quad y = \begin{pmatrix} | & | & | \\ y_1 & y_2 & y_3 \\ | & | & | \end{pmatrix} \implies z = \begin{pmatrix} x_1 \cdot y_1 & \dots & x_1 \cdot y_3 \\ x_2 \cdot y_1 & \dots & \dots \\ \dots & \dots & \dots \end{pmatrix}$$

支持使用 `...` 匹配任意前置维度（如 batch 维度）：

```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

### 4.3 `einops.reduce()`
* **`reduce`**: 按指定的维度模式，把某些维度通过 `sum`、`mean`、`max`、`min` 等操作消除掉。

```python
x = torch.ones(2, 3, 4)  # batch seq hidden

# Old way:
y = x.sum(dim=-1)

# New way:
y = reduce(x, "... hidden -> ...", "sum")
```

* **直观示意**：将长度为 $4$ 的 `hidden` 维度（元素全为 $1$）求和消去，得到形状为 $(2, 3)$、元素全为 $4$ 的张量：
  $$\begin{bmatrix} 4 & 4 & 4 \\ 4 & 4 & 4 \end{bmatrix}$$

### 4.4 `einops.rearrange()`

```python
x = torch.ones(3, 8)  # seq total_hidden
# where total_hidden is a flattened representation of heads * hidden1

# Break up total_hidden into two dimensions (heads and hidden1)
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
# x.shape = (3, 2, 4)

w = torch.ones(4, 4)  # hidden1 hidden2

# Perform the transformation by w
x = einsum(x, w, "... heads hidden1, hidden1 hidden2 -> ... heads hidden2")

# Combine heads and hidden2 together
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

---

## 5. Tensor Operations FLOPs (张量运算与浮点计算量)

### 5.1 FLOPs 与 FLOP/s 的区别
A floating-point operation (FLOP) is a basic operation like addition ($x + y$) or multiplication ($xy$).
* **FLOPs**: floating-point operations，表示**计算量**。
* **FLOP/s**: floating-point operations per second (also written as FLOPS)，衡量**硬件性能**。

### 5.2 Linear Model 的计算量
Suppose we have a linear model:
* We have $n$ (or $B$) points.
* Each point is $d$-dimensional (or $D$).
* The linear model maps each $d$-dimensional vector to $k$ outputs (or $K$).

```python
x = torch.ones(B, D, device=cuda_if_available())
w = torch.ones(D, K, device=cuda_if_available())
y = x @ w
actual_num_flops = 2 * B * D * K
```

* **推导说明**：计算输出矩阵中的单个元素时，需进行 $D$ 次乘法和 $D - 1$ 次加法，此处近似为 $2D$ 次浮点运算；输出矩阵共有 $B \times K$ 个元素，故总计算量为 $2 \times B \times D \times K$。
* **Interpretation**:
  * $B$ is the number of data points.
  * $(D \times K)$ is the number of parameters.
  * **FLOPs for forward pass is $2 \times (\text{\# tokens}) \times (\text{\# parameters})$**.
  * It turns out this generalizes to Transformers (to a first-order approximation).

### 5.3 测量实际耗时与 MFU
*How do FLOPs calculations translate to wall-clock time (seconds)? Let us time it!*

```python
actual_time = time_matmul(x, w)  # matrix multiplication
actual_flop_per_sec = actual_num_flops / actual_time

def time_matmul(a: torch.Tensor, b: torch.Tensor) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    def run():
        a @ b
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)
    return total_time / num_trials
```

Each GPU has a specification sheet that reports the peak performance.
*(Note that the FLOP/s depends heavily on the data type!)*

```python
promised_flop_per_sec = get_promised_flop_per_sec(x.dtype)
```

**Model FLOPs Utilization (MFU)**:
$$\text{MFU} = \frac{\text{actual FLOP/s}}{\text{promised FLOP/s}} \quad \text{[ignore communication/overhead]}$$

---

## 6. Arithmetic Intensity (算术强度 / 运算强度)

### 6.1 硬件执行计算的步骤与耗时
**How to compute a thing:**
1. Send inputs from memory to accelerator.
2. Perform computation.
3. Send outputs from accelerator to memory.

**How long does it take? It depends on two things:**
1. **Accelerator speed** ($\text{FLOP/s}$)
2. **Memory bandwidth** ($\text{byte/s}$)

```python
assert h100_flop_per_sec == 1979e12 / 2  # Half without sparsity
assert h100_bytes_per_sec == 3.35e12
```

### 6.2 案例一：ReLU 的算术强度
**Rectified Linear Unit (线性整流单元)**：$f(x) = \max(0, x)$

```python
def arithmetic_intensity_relu():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = torch.relu(x)
    bytes = (2 * n) + (2 * n)  # Read x, Write y (需移动的字节数)
    flops = n  # n comparisons
    communication_time = bytes / h100_bytes_per_sec  # = 1.252e-6
    computation_time = flops / h100_flop_per_sec     # = 1.060e-9
    # 通常通信与计算是可以同时进行的 (overlap)
    total_time = max(communication_time, computation_time)
```

### 6.3 性能瓶颈判定 (Memory-bound vs. Compute-bound)
**What is the bottleneck?**
* **Memory-bound**: $\text{communication\_time} > \text{computation\_time}$
* **Compute-bound**: $\text{communication\_time} < \text{computation\_time}$

In this case, ReLU is **memory-bound**.

**Alternative way to see this:**

| 指标名称 | 物理意义 | 计算公式 |
| :--- | :--- | :--- |
| **Accelerator intensity** | How much work can the accelerator do per byte transferred | $\text{h100\_flop\_per\_sec} / \text{h100\_bytes\_per\_sec}$ |
| **Arithmetic intensity** | How much actual work is performed per byte for this workload（计算任务/负载，指交给计算机系统去执行的具体计算工作量） | $\text{flops} / \text{bytes}$ |

* **Memory-bound**: $\text{arithmetic\_intensity} < \text{accelerator\_intensity}$
* **Compute-bound**: $\text{arithmetic\_intensity} > \text{accelerator\_intensity}$

```python
assert arithmetic_intensity < h100_accelerator_intensity
```

> **注**：Python 中，`assert` 是一个关键字/语句（而不是内置函数），用于在代码中声明某个条件必为真。若条件为假，程序会立即抛出 `AssertionError` 异常并中断执行，常用于内部调试与测试。

In general, we'll find ourselves memory-bound, so **higher arithmetic intensity is good**（避免大量时间耗费在转移数据上）.

### 6.4 案例二：向量点积与矩阵乘法

```python
def arithmetic_intensity_dot_product():
    n = 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w
    bytes = (2 * n) + (2 * n) + 2
    flops = 2 * n - 1
    arithmetic_intensity = flops / bytes  # ~ 1/2 (笔记手写标注为 n/2)
    # Memory-bound!
```

同理可算得：
* **`arithmetic_intensity_matrix_vector_product`** $\rightarrow$ **Memory-bound**
* **`arithmetic_intensity_matmul`** $\approx \frac{n}{3} \rightarrow$ **Compute-bound**（当 $n$ 较大时）
  * **解释了为何会使用大规模矩阵**：因为 $n$ 较小时，如果你没有达到加速器强度（Accelerator intensity），实际上不会让速度更快。
  * Training Transformers involves big matrix multiplications.
  * Matrix-vector product is what happens during inference, which is why **inference is memory-bound**.

### 6.5 屋顶线图 (Roofline Plots)

```python
def roofline_plots():
    ...
```

* **坐标轴说明**：
  * **横轴 (X-axis)**：Arithmetic Intensity ($\text{FLOPs/byte}$)
  * **纵轴 (Y-axis)**：Performance ($\text{FLOP/s}$)，上限受限于水平线 **Accelerator peak FLOP/s**
  * **斜线**：代表不同内存带宽限制下的性能上限（例如带宽较低的 $\text{BW}_1$ 与带宽较高的 $\text{BW}_2$）
* **算法状态分析**：
  * **$\text{Algo}_1$（低算术强度）**：在 $\text{BW}_1$ 和 $\text{BW}_2$ 下均处于斜坡区域 $\rightarrow$ **Bandwidth Bound at $\text{BW}_1, \text{BW}_2$**
  * **$\text{Algo}_2$（中等算术强度）**：在低带宽 $\text{BW}_1$ 下受限于斜坡，在高带宽 $\text{BW}_2$ 下触及水平屋顶 $\rightarrow$ **Bandwidth Bound at $\text{BW}_1$, Compute Bound at $\text{BW}_2$**

---

## 7. Deep Linear Network 示例

```python
def deep_linear_network():
    # Consider a deep network with L layers and D-dimensional
    # inputs, activations and outputs.
    # Define the network
    D = 8
    L = 4
    model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())
    num_parameters = get_num_parameters(model)
    assert num_parameters == (D * D) * L

    # Run the model on a batch of data
    B = 4  # Batch size
    x = torch.randn(B, D, device=cuda_if_available())
    y = model(x)
```                                                                                                                           
注：前面这部分内容本来是我记在纸质笔记本上的，后来我使用gemini4 argon模型将原纸质笔记对应的pdf文档直接识别并转换成.md格式的。若对转化结果有任何疑问，可参考原pdf文件。下面所有内容都是笔者直接用.md文件记录的。                                        
```python
def deep_linear_network()

### define the model
D=8
L=3
model=DeepNetwork(dim=D,num_layers=L).to(cuda_if_avaliable())

num_parameter=get_num_parameters(model)
assert num_parameters == D*D*L

# Run the model

```

```python
class DeepNetwork(nn.Module): # 从DeepNetwork是从父类继承来的
""" Map `dim`-vector to a `dim`-vector """

    def __init__(self, dim:int, num_layers:int): #定义构造函数，每次生成新实例时会自动调用这个构造函数
        super().__init__()  #调用父类的构造函数，因为当前类是从父类继承来的，所以得先让父类把底层的系统机制初始化好，后面的模型才能正常工作
        self.layers=nn.ModuleList([Block(dim) for i in range(num_layers) ]) #给实例添加layer属性，类属性后面是没括号的，类方法后面是有括号的

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x=layer(x) #layer虽然是一个对象，但是python中有一个可调用对象
                       #的机制，只要在类内部定义了__call__方法，那么这个
                       #类造出的的实例对象就可以像函数一样加括号调用
        return
```

```python
class Block(nn.module):
    """Simple block that applies a linear transformation followed by a Relu nonlinearly """

    def __init__(self,dim:int):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(dim,dim)/np.sqrt(dim)) #nn会将普通张量包装成PyTorch可训练参数
    def forward(self,x:torch.tensor) ->torch.tensor:
        x=x@self.weight 
        x=F.relu(x)
        
        return x
```

```python
def gradients_basics():
    #So far, We constructed tensors and passed them through operations(forward).Now,we're going to compute the gradients(gradients).

    #As a simple example, let's consider the simple linear model:
    y=0.5*(x*w-5)^2

    #Forward pass:compute loss
    x=torch.tensor([1.,2.,3.]) #1.的数据类型是浮点数，1的数据类型是整数，神经网络中的张量元素更通常用浮点数来表示（PyTorch 自动微分只支持浮点数和复数类型的张量。）
    w=torch.tensor([1.,1.,1.],requires_grad=True)#告诉PyTorch，在计算过程中，跟踪与w有关的可微运算，以便计算后续Loss对w的梯度。
    pred_y=x@w
    loss=0.5*(pred_y-5).pow(2)
    #Backward pass: compute gradients
    loss.backward()
    
    #默认情况下，PyTorch只会把梯度累积到需要梯度的叶子张量的.grad属性中。
    assert loss.grad is None
    assert pred_y.grad is None
    assert x.grad is None
    assert torch.equal(w.grad, torch.tensor([1.,2.,3.]))

```
下面计算一下计算梯度的过程需要多少计算量：

```python
def gradients_flops():

    #考虑一个简单的例子（一个双层线性神经网络，且暂时忽略Relu）
    B=1024 #number of points
    D=256  #number of Dimension

    x=torch.ones(B，D, device=cuda_if_avaliable)
    w1=torch.randn(D, D, device=cuda_if_avaliable)
    w2=torch.randn(D, D, device=cuda_if_avaliable)
    # forward
    h1=einsum(x, w1, "batch in, in out -> batch out") 
    h2=einsum(h1, w2, "batch in, in out -> batch out")
    loss= (h2.mean()-0)**2  #Regress everthing to a 0 (arbitrary)
    # backward
    h1.retain_grad()  #保留loss对h1的梯度（for debugging）
    h2.retain_grad()  #for debugging
    loss.backward()

    # Zoom in one layer (focus on layer h1 --w2 --> h2)
    # Forward pass: (bf16)
    num_flops= B*D*2*D
    # Backward pass: h1.grad= dloss/dh1; w2.grad= dloss/dw2;
    h1_grad= einsum(h2.grad, w2, "batch out, in out -> batch in" )
    assert torch.allclose(h1.grad, h1_grad)
    w2_grad= einsum(h2.grad, h1, "batch out, batch in -> in out")
    assert torch.allclose(w2.grad, w2_grad)

    num_backward_flops= (2*B*D*D) + (2*B*D*D) #注意到反向传播的计算量是前向传播的两倍，因为级要计算输入(此处是h1，计算h1是更新前面层参数必须的一步)的梯度 + 参数的梯度

    #以上的计算步骤只针对w2。
```

Consider all layers:
* Forward pass: 2(#data points)(#parameters)**2 FLOPs
* Backward pass: 4(#data points)(#parameters)**2 FLOPS
* Total: 6(#data points)(#parameters)**2 FLOPS

This is for MLPs...but it turns out to be a good approximation for Transformation for short context lengths as well.(如果上下文太长，计算量就变成了上下文的平方)


Let's define the AdaGrad optimizer:
* momentum = SGD + exponential averraging of grad
* AdaGrad = SGD + averaging by grad^2
* RMSProp = AdaGrad + exponentially averaging of grad^2
* Adam = RMSProp + momentum

补充知识：
* SGD(stochastic gradient descent): $\theta_{t+1}=\theta_t - \eta g_t$（这是其他优化方法的基础）
缺点：全局使用一个学习率$\eta$，若损失曲面呈狭长峡谷，梯度会在两侧剧烈震荡，沿着峡谷底部的推进及其缓慢。
* momentum（全局学习率不变，但是更新方向不再看当前单步的梯度，而是看历史梯度的滑动平均$m_t$）: 
$$ m_t=\beta m_{t-1}+(1-\beta) g_t, \space \beta \approx 0.9 $$
$$  \theta_{t+1} = \theta_{t} + \eta m_t  $$
    作用：在震荡方向上，正负梯度相互抵消，振幅显著减小。在一致下降方向，动量会持续积累。

* AdaGrad: 不再对梯度方向做平滑，而是针对每个参数，根据其历史梯度的平方和来自适应调整其独立的有效学习率：
$$v_t=v_{t-1} + g_t^2 $$
$$\theta_{t+1} = \theta_t + \frac{\eta}{\sqrt{v_t+\epsilon}} \odot g_t$$
    上式中$\odot$表示哈达玛积，因为此时的$\eta$是一个矢量，不同的分量对应于不同的参数，也就是说，每个参数对应的学习率是不一样的。但前两种优化方法中，学习率是一个全局常量。
    作用：
    * 频繁更新/梯度很大的参数：$v_t$ 迅速增大，其有效学习率被压低，防止步子迈得过大。
    * 稀疏特征/梯度较小的参数：$v_t$ 累积缓慢，保留较大的有效学习率，加快学习。

    缺点：
    * $v_t$ 是纯单调递增的累加。随着训练步数增加，分母无限膨胀，有效学习率过早衰减至接近 0，导致模型在收敛前“提前停滞”。

* RMSProp(Root Mean Square Propagation,由 Geoffrey Hinton 提出，核心目的是解决 AdaGrad 学习率单调递减至死的问题。):
$$ v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \space \beta_2 \approx 0.99$$
$$\theta_{t+1} = \theta_t + \frac{\eta}{\sqrt{v_t+\epsilon}} \odot g_t $$

    作用：
    * 引入遗忘系数 $\beta_2$（通常取 0.99）。$v_t$ 代表的是近期梯度平方的加权平均，不再无限制单调累加。
    * 保留了 AdaGrad 对不同参数自适应缩放学习率的优势。摆脱了历史遥远梯度的束缚，有效学习率能够动态适应当前阶段的损失曲面变化。

* Adam（Adaptive Moment Estimation）将 Momentum（一阶动量） 和 RMSProp（二阶动量） 结合在一起，并引入偏差修正：
$$一阶矩（动量）：m_t = \beta_1 m_{t-1}+(1-\beta_1)g_t$$
$$ \text{二阶矩（RMSProp）：} \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\text{偏差修正：} \quad \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\text{参数更新：} \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$
AdaGrad [Duchi+2011]
```python
optimizer = AdaGrad(model.parameters(),lr=0.01)
state = model.state_dict() #返回一个 Python 字典（OrderedDict），包含模型每一层的名字及其对应的参数张量（权重和偏置）与持久缓存（如 BatchNorm 的均值/方差），常用于模型保存或权重查看。

# Computer gradients
x = torch.randn(B, D, device=cuda_if_avaliable)
y = torch.tensor([4.,5.], device =cuda_if_avaliable)
pred_y = model(x).mean() #model(x)：调用模型的 __call__ 方法执行前向传播（Forward Pass）。
                         #.mean()：对模型输出的所有元素求均值，将输出压缩为一个标量（Scalar）。
loss = F.mse_loss(input=pred_y, target=y)
loss.backward()

# teke a step
optimizer.step() #作用：根据优化算法更新模型参数。Adagrad 会结合历史梯度的平方累积和，调整每个参数的步长
optimizer_state = {i: dict(p_state) for i, (p,p_state) in enumerate(optimizer.state.items())} #optimizer.state 是一个默认字典（defaultdict），键为参数对象 Parameter，值为该参数维护的优化历史状态（例如 Adagrad 维护的累计梯度平方和 sum、步数 step 等）。enumerate(...)：给遍历的元素打上自增整数索引：(0, (p_0, p_state_0)), (1, (p_1, p_state_1)), ...。{i: ... for ...}：将整数序号 i 作为新字典的键。

# free the memory
optimizer.zero_grad(set_to_none=True) #set_to_none=True：不将 .grad 赋值为全零张量，而是直接设为 None。这样能释放 .grad 占用的显存，并在下一次计算梯度时省去清零操作，提升运行效率。
```

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, parameters: Iterable[nn.Parameter], lr:float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get squared gradients g2 = sum_{i<t}g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))

                # Update  optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2

                # Update parameters
                p.data -=lr * grad/torch.sqrt(g2 + 1e-5)
                
```

**Memory**

接下来，讨论整体的内存占用情况（依然讨论上述神经网络架构）。
```python
# parameter
parameter_memory = 2*(D*D*L)  # 2 bytes for bf16。这里的模型是每一层神经元数量等于输入维度，实际情况二者不一定相等

# activation（激活值就是每一层每个神经神经元的输出值，这些值会作为下一层的输入）
activition_memory = 2*B*D*L  #2 bytes for bf16

# Gradients
gradient_memory = 2* parameter_number   # 2 bytes for bf16 

# Optimizer state
optimizer_ state_memory = 4 * parameter_number  # 4 bytes for bf32. Customary to use fp32 for stability(accumulating averages over powers over many steps), 4 bytes/parameters for storing second moments(AdaGrad). Adam requires 8 bytes/parameter for storing first and second moments.  

``` 
内存（在深度学习中主要是 GPU 的显存/HBM(High Bandwidth Memory )）从两个截然不同的物理属性上，同时卡住了大模型的“空间”与“时间”：
* 显存容量（Capacity，以 GB 为单位）：决定了能存下多大的模型（能不能跑）。
* 显存带宽（Bandwidth，以 GB/s 或 TB/s 为单位）：在大量场景下直接决定了计算速度有多快（跑得快不快）。

*简单介绍一下节省内存的两种方法**

1. Gradient accumulation:
* compute gradient on micro batcher
* accumulate the gradients(don't zero it out)
* every batch_size/micro_size steps, upgrade the parameters and zero out the gradients 
```python
def gradient_accumulation():
    #large batch size: improve training stability. However, activition memory scales with batch size, so might run out.
    B=1024
    L=16
    D=1024
    activtion_memory= 2*B*D*L
```
2. ctivtion_checkpointing():
```python
def activition_checkpointing():
    #for training, we need to store the activations of all layers;
    # for inference, we don't compute gradients, so we only need to store the current layer's activitions. 
    # 具体实现方法此处不再叙述。
```


