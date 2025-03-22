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
