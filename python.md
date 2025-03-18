import sys
input=sys.stdin.readline # 读入会变快


## 检查数字是否能用连续正整数相加表示：
一个数 𝑥 能表示为至少两个连续正整数之和，当且仅当它不是 2 的幂。因为 2 的幂无法拆分成两个或以上的连续正整数之和。

## [回文字符串](https://www.lanqiao.cn/problems/19718/learning/?page=1&first_category_id=1&tags=%E5%89%8D%E7%BC%80%E5%92%8C&tag_relation=union&sort=difficulty&asc=1)

```python
n=eval(input())
L=[]
for i in range(n):
  s=input()
  for j in s:
    if j in ['q','b','l']:
      s=s.replace(j,'')
  if s==s[::-1]:
    print('Yes')
  else:
    print('No')
# 请在此输入您的代码
```

```python
import os
import sys

# 请在此输入您的代码
n = int(input())
for _ in range(n):
    s = input().strip()  # 去掉首尾空格
    filtered_s = ''.join(c for c in s if c not in ['l', 'b', 'q'])  # 过滤字符
    if filtered_s == filtered_s[::-1]:  # 判断是否回文
        print("Yes")
    else:
        print("No")
```
- [寻找AKKO](https://www.lanqiao.cn/problems/3907/learning/?page=1&first_category_id=1&tags=%E5%89%8D%E7%BC%80%E5%92%8C&tag_relation=union&sort=difficulty&asc=1)

```bash
import os
import sys

n = int(input())  # 读取字符串长度
s = str(input())  # 读取字符串

# 计数变量
count_A = 0   # 统计字符 'A' 的数量
count_K1 = 0  # 统计 "AK" 形式的数量
count_K2 = 0  # 统计 "AKK" 形式的数量
count_O = 0   # 统计 "AKKO" 子序列的数量

# 遍历字符串
for i in s:
    if i == 'A':
        count_A += 1  # 统计 'A' 的数量
    if i == 'K':
        count_K2 += count_K1  # "AKK" 的数量增加
        count_K1 += count_A   # "AK" 的数量增加
    if i == 'O':
        count_O += count_K2  # "AKKO" 的数量增加

print(count_O)  # 输出 "AKKO" 子序列的数量
```

- [二维前缀和](https://www.lanqiao.cn/problems/18439/learning/?page=1&first_category_id=1&tags=%E5%89%8D%E7%BC%80%E5%92%8C&tag_relation=intersection&sort=pass_rate&asc=0)
```python
import sys

# 读取输入
n, m, q = map(int, input().split())
a = [[0] * (m + 1) for _ in range(n + 1)]
s = [[0] * (m + 1) for _ in range(n + 1)]

# 构建前缀和
for i in range(1, n + 1):
    row = list(map(int, input().split()))
    for j in range(1, m + 1):
        a[i][j] = row[j - 1]
        s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + a[i][j]

# 处理查询
for _ in range(q):
    x1, y1, x2, y2 = map(int, input().split())
    result = s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 - 1] + s[x1 - 1][y1 - 1]
    print(result)
```

- [其他元素的乘积](https://www.lanqiao.cn/problems/317/learning/?page=1&first_category_id=1&tags=%E5%89%8D%E7%BC%80%E5%92%8C&tag_relation=intersection&sort=pass_rate&asc=0)

```bash
import sys

# 读取输入
n = int(input())  # 数组大小
a = list(map(int, input().split()))  # 读取数组元素

# 计算前缀积
prefix = [1] * n
for i in range(1, n):
    prefix[i] = prefix[i - 1] * a[i - 1]

# 计算后缀积
suffix = [1] * n
for i in range(n - 2, -1, -1):
    suffix[i] = suffix[i + 1] * a[i + 1]

# 计算最终结果
result = [prefix[i] * suffix[i] for i in range(n)]

# 输出
print(" ".join(map(str, result)))
```

- [一维前缀和](https://www.lanqiao.cn/problems/18437/learning/?page=1&first_category_id=1&tags=%E5%89%8D%E7%BC%80%E5%92%8C&tag_relation=intersection&sort=pass_rate&asc=0)

```python
import os
import sys

# 请在此输入您的代码
n,q=map(int,input().split())
a=list(map(int,input().split()))
ans=[0]*(n+1)
for i in range(1,n+1):
  ans[i]=ans[i-1]+a[i-1]
for i in range(q):
  l,r=map(int,input().split())
  print(ans[r]-ans[l-1])
```
