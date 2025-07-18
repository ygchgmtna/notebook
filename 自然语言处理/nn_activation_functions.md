
# PyTorch 激活函数详解：Softmax、Sigmoid、Tanh、LeakyReLU、ELU 等

本文汇总介绍 PyTorch 中常见的激活函数模块，包括其 **数学定义、适用场景、优缺点、代码示例与对比总结**。

---

## ✅ 一、`nn.Sigmoid`

### 数学定义：
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

### 特点：
- 输出范围：\( (0, 1) \)
- 常用于二分类任务最后一层输出
- 缺点：容易饱和、梯度消失

### 示例：
```python
import torch.nn as nn
sigmoid = nn.Sigmoid()
x = torch.tensor([-1.0, 0.0, 1.0])
y = sigmoid(x)  # 输出约为 [0.27, 0.5, 0.73]
```

---

## ✅ 二、`nn.Softmax`

### 数学定义（对向量 \(x\)）：
```math
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
```

### 特点：
- 输出是一个概率分布，所有元素相加为 1
- 常用于多分类任务的最后一层
- 通常与 `nn.CrossEntropyLoss` 不一起使用（该损失内部已包含 softmax）

### 示例：
```python
softmax = nn.Softmax(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0]])
y = softmax(x)  # 输出：[0.09, 0.24, 0.66]
```

---

## ✅ 三、`nn.Tanh`

### 数学定义：
```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

### 特点：
- 输出范围：\( (-1, 1) \)
- 类似 sigmoid，但中心对称
- 仍可能出现梯度消失问题

### 示例：
```python
tanh = nn.Tanh()
x = torch.tensor([-1.0, 0.0, 1.0])
y = tanh(x)  # 输出约为 [-0.76, 0.0, 0.76]
```

---

## ✅ 四、`nn.LeakyReLU`

### 数学定义：
```math
f(x) = \begin{cases}
x, & x > 0 \\
\alpha x, & x \leq 0
\end{cases}
```

### 特点：
- 允许负值通过，缓解“ReLU 死亡”问题
- `alpha`（默认 0.01）是负斜率

### 示例：
```python
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
x = torch.tensor([-1.0, 0.0, 1.0])
y = leaky_relu(x)
```

---

## ✅ 五、`nn.ELU`（Exponential Linear Unit）

### 数学定义：
```math
f(x) = \begin{cases}
x, & x > 0 \\
\alpha (e^x - 1), & x \leq 0
\end{cases}
```

### 特点：
- 比 LeakyReLU 更平滑
- 输出均值更接近 0，提升收敛速度

### 示例：
```python
elu = nn.ELU(alpha=1.0)
x = torch.tensor([-1.0, 0.0, 1.0])
y = elu(x)
```

---

## ✅ 六、`nn.Softplus`

### 数学定义：
```math
f(x) = \log(1 + e^x)
```

### 特点：
- 平滑近似于 ReLU
- 输出总为正
- 可微、适合于某些需要连续梯度的模型

### 示例：
```python
softplus = nn.Softplus()
x = torch.tensor([-1.0, 0.0, 1.0])
y = softplus(x)
```

---

## 📊 总结对比表

| 激活函数       | 输出范围     | 是否中心对称 | 常见用途                   | 缺点                   |
|----------------|---------------|---------------|----------------------------|------------------------|
| `Sigmoid`      | (0, 1)         | 否            | 二分类输出层               | 梯度消失，饱和         |
| `Softmax`      | (0, 1), 和为1  | 否            | 多分类输出层               | 不适用于中间层         |
| `Tanh`         | (-1, 1)        | 是            | RNN中常用，增强表达力      | 梯度消失               |
| `ReLU`         | [0, ∞)         | 否            | CNN中常用，速度快          | 死亡ReLU问题           |
| `LeakyReLU`    | (-∞, ∞)        | 否            | 深层CNN，防止神经元死亡    | 有负斜率超参           |
| `ELU`          | (-α, ∞)        | 否            | 深层网络，稳定训练         | 计算稍慢               |
| `Softplus`     | (0, ∞)         | 否            | 平滑ReLU替代               | 输出偏大，不居中       |

---

如你还想了解 Swish、GELU、SiLU、Mish 等近年常见的激活函数，我也可以继续扩展文档内容。
