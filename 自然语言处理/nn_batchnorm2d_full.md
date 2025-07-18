
# PyTorch `nn.BatchNorm2d()` 全面介绍

本文详细介绍 PyTorch 中的 `nn.BatchNorm2d` 模块，包括其 **作用、原理、参数、运行机制、使用示例、训练与推理模式的区别、与 LayerNorm/InstanceNorm 对比等**。

---

## ✅ 一、`nn.BatchNorm2d()` 是什么？

`nn.BatchNorm2d` 是 PyTorch 中用于 **二维输入特征图（如图像）** 的 **批量归一化层**。它在每个小批量数据中，对每个通道（channel）进行独立的标准化处理，缓解了深层神经网络训练中的梯度消失/爆炸等问题。

---

## 🧠 二、Batch Normalization 的基本原理

BN 的核心思想是在每个 mini-batch 内，使每个通道的特征服从 **标准正态分布**（均值为 0，方差为 1），并引入可学习参数进行恢复调整。

### 数学表达式：

对于输入特征图 `x`，其维度为 `(N, C, H, W)`，BN 的处理为：

```math
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} \\
y_{nchw} = \gamma_c \cdot \hat{x}_{nchw} + \beta_c
```

- \( \mu_c \)：通道维度上的均值（在一个 mini-batch 中）
- \( \sigma_c^2 \)：通道维度上的方差
- \( \gamma_c, \beta_c \)：可学习的缩放与平移参数
- \( \epsilon \)：防止除以 0 的小常数

---

## 🔧 三、函数定义与参数

```python
torch.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
```

| 参数名                | 含义 |
|-----------------------|------|
| `num_features`        | 输入的通道数 `C` |
| `eps`                 | 加到方差上的小数，避免除 0 |
| `momentum`            | 更新运行均值/方差的动量系数 |
| `affine`              | 是否学习 `γ` 和 `β` |
| `track_running_stats` | 是否记录运行时均值与方差（用于推理） |

---

## 📘 四、使用示例

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm2d(num_features=16)

# 输入：[batch_size, channels, height, width]
x = torch.randn(8, 16, 32, 32)
y = bn(x)

print(y.shape)  # torch.Size([8, 16, 32, 32])
```

---

## 🔍 五、训练模式 vs 推理模式

### ✅ 训练模式（`model.train()`）：
- 使用当前 batch 的均值和方差
- 同时更新 `running_mean` 和 `running_var`

### ✅ 推理模式（`model.eval()`）：
- 使用之前训练时记录的 `running_mean` 和 `running_var`
- 保证预测稳定性

切换方式：

```python
model.train()  # 训练模式
model.eval()   # 推理模式
```

---

## 🧪 六、参数说明（内部变量）

```python
bn.running_mean  # shape: [C]
bn.running_var   # shape: [C]
bn.weight        # γ，可学习参数
bn.bias          # β，可学习参数
```

这些参数会在训练时自动更新，并在推理时使用。

---

## 🔁 七、与其他归一化层的对比

| 层类型          | 归一化维度      | 是否依赖 batch 大小 | 用途                          |
|-----------------|------------------|----------------------|-------------------------------|
| `BatchNorm2d`   | 每个通道         | ✅ 是                | CNN 中常用                   |
| `LayerNorm`     | 每个样本的所有特征 | ❌ 否               | NLP, Transformer              |
| `InstanceNorm2d`| 每个样本每通道   | ❌ 否                | 图像生成、风格迁移等         |
| `GroupNorm`     | 每组通道         | ❌ 否                | 比 BN 更稳定，适用于小 batch |

---

## 🎯 八、典型用法示例（卷积+BN+ReLU）

```python
block = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU()
)
```

这种结构是 ResNet 等网络的基础构件，有助于加速收敛、提高稳定性。

---

## ✅ 九、总结

| 点位         | 内容说明                            |
|--------------|-------------------------------------|
| 本质         | 对每个通道做归一化，控制分布形态     |
| 关键特性     | 缓解梯度问题，稳定训练过程           |
| 推理模式     | 使用运行时统计量（均值与方差）       |
| 可学习参数   | 缩放因子 γ 与 平移偏置 β             |
| 常见搭配     | 与卷积层+激活函数组合使用            |

---

如你希望深入了解 BN 的反向传播公式、与小批量策略兼容性或在 Transformer 中的替代方式（如 LayerNorm），也欢迎继续提问！
