
# PyTorch `nn.Linear()` 全面介绍

下面是对 `nn.Linear()` 的详细介绍，涵盖其 **作用、参数、原理、示例、初始化机制、与 Conv2d 的关系** 等内容。

---

## ✅ 一、`nn.Linear()` 是什么？

`nn.Linear()` 是 PyTorch 中用于定义 **全连接层（Fully Connected Layer）** 或称 **线性层** 的函数。

它实现了如下的线性变换：

```math
\text{output} = x \cdot W^T + b
```

- \( x \)：输入张量，形状为 `(batch_size, in_features)`
- \( W \)：权重矩阵，形状为 `(out_features, in_features)`
- \( b \)：偏置项，形状为 `(out_features,)`

---

## 🔧 二、函数定义与参数

```python
torch.nn.Linear(in_features, out_features, bias=True)
```

| 参数           | 说明                                     |
|----------------|------------------------------------------|
| `in_features`  | 输入特征数（输入张量的最后一维大小）     |
| `out_features` | 输出特征数（线性层的输出大小）           |
| `bias`         | 是否包含偏置项，默认为 `True`            |

---

## 📦 三、权重和偏置

创建 `nn.Linear` 层后，自动包含：

- `weight`: `[out_features, in_features]`
- `bias`: `[out_features]`（若设置为 `True`）

```python
linear = nn.Linear(3, 2)
print(linear.weight.shape)  # torch.Size([2, 3])
print(linear.bias.shape)    # torch.Size([2])
```

---

## 🧠 四、工作原理（forward过程）

```python
output = input @ weight.T + bias
```

其中 `@` 表示矩阵乘法。

---

## 📘 五、简单示例

```python
import torch
import torch.nn as nn

fc = nn.Linear(3, 2)
x = torch.tensor([[1.0, 2.0, 3.0]])
y = fc(x)
print(y)  # shape: (1, 2)
```

---

## 🔍 六、常见应用场景

| 场景                   | 用法                                 |
|------------------------|--------------------------------------|
| 神经网络分类器的输出层 | `nn.Linear(hidden_dim, num_classes)` |
| Transformer 的 Q/K/V  | `nn.Linear(embed_dim, head_dim)`     |
| 编码特征映射           | 改变维度，如 `nn.Linear(768, 256)`    |

---

## 🧪 七、是否带偏置的区别

```python
nn.Linear(128, 64, bias=True)
nn.Linear(128, 64, bias=False)
```

不加偏置常用于搭配 BatchNorm 层，以避免参数冗余。

---

## 🧰 八、配合激活函数使用

线性层本身不包含激活函数，通常搭配使用：

```python
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

---

## 🧬 九、查看权重和梯度

```python
fc = nn.Linear(3, 2)
out = fc(torch.randn(1, 3))
loss = out.sum()
loss.backward()

print(fc.weight.data)
print(fc.weight.grad)
```

---

## ✅ 十、总结

| 点位     | 说明                              |
|----------|-----------------------------------|
| 本质     | 线性映射 `y = xW^T + b`           |
| 应用     | 分类器、注意力投影、特征变换等    |
| 参数     | `in_features`, `out_features`, `bias` |
| 配合使用 | 与激活函数、BatchNorm 搭配使用   |

---

如需了解其源码机制、初始化细节或与 `Conv2d` 的区别，请参考另一篇深入讲解。
