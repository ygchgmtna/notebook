## np.means()
np.mean() 是 NumPy 提供的用于计算平均值的函数。
```python
np.mean(a, axis=None, dtype=None, keepdims=False)
```
```
a	要计算平均值的数组
axis	指定沿哪个轴计算平均值（默认是全部）
dtype	指定计算时的数据类型
keepdims	是否保留原数组的维度
```

### 举例说明

```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
```
[[1, 2], 
 [3, 4]]
```python
# 不指定 axis：计算所有元素的平均值
np.mean(a)
# 结果：2.5

# 指定 axis=0：按列求平均
np.mean(a, axis=0)
# 结果：[2. 3.]

# 指定 axis=1：按行求平均
np.mean(a, axis=1)
# 结果：[1.5 3.5]
```
keepdims=True 用法（保持原维度）
```python
np.mean(a, axis=1, keepdims=True)
# 结果：
[[1.5]
 [3.5]]
```
你会看到结果是列向量 (2,1)，而不是压扁成一维数组。方便和原数据一起做矩阵运算。

## 协方差矩阵

```python
np.cov(m, rowvar=True, bias=False, ddof=None)
```

### 参数解释

```python
m	输入数组，形状为 (n_features, n_samples)（默认），或 (n_samples, n_features) 配合 rowvar=False
rowvar	是否每一行代表一个变量（默认是 True）。如果你的变量按列排列（常见），需要设为 False
bias	是否使用偏差估计（除以 N 而不是 N-1）。设为 True 使用偏差估计
ddof	自由度调整参数，默认是 1（即除以 N-1）
返回值	协方差矩阵（二维数组），形状为 (n_features, n_features)
```

### 举例说明

```python
import numpy as np

# 样本为行，特征为列（常见形式）
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# 计算协方差矩阵
cov_matrix = np.cov(X, rowvar=False)
print(cov_matrix)
# 输出
[[4. 4.]
 [4. 4.]]
# 说明
X 有两个特征（列），每个特征的方差是 4；
两个特征之间的协方差是 4，说明它们是正相关的。
```

### 行为变量 vs 列为变量

默认情况下 rowvar=True，意味着：
```
每一行是一个变量，每一列是一个观测值（不常见）
```
如果我们像上面那样：
```
每一行是一个样本，每一列是一个变量（最常见）
```
就应该加上：
```python
np.cov(X, rowvar=False)
```
