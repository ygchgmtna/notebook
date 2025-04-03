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

- [区间次方和](https://www.lanqiao.cn/problems/3382/learning/?page=1&first_category_id=1&tags=%E5%89%8D%E7%BC%80%E5%92%8C&tag_relation=intersection&sort=pass_rate&asc=0)

```python
import os
import sys

# 请在此输入您的代码

n,m=map(int,input().split())
a=list(map(int,input().split()))
ans1=[0]*(n+1)
ans2=[0]*(n+1)
ans3=[0]*(n+1)
ans4=[0]*(n+1)
ans5=[0]*(n+1)
for i in range(1,n+1):
  ans1[i]=ans1[i-1]+a[i-1]**1
  ans2[i]=ans2[i-1]+a[i-1]**2
  ans3[i]=ans3[i-1]+a[i-1]**3
  ans4[i]=ans4[i-1]+a[i-1]**4
  ans5[i]=ans5[i-1]+a[i-1]**5
for i in range(m):
  l,r,k=map(int,input().split())
  if k==1:
    print((ans1[r]-ans1[l-1])%(10**9+7))
  elif k==2:
    print((ans2[r]-ans2[l-1])%(10**9+7))
  elif k==3:
    print((ans3[r]-ans3[l-1])%(10**9+7))
  elif k==4:
    print((ans4[r]-ans4[l-1])%(10**9+7))
  elif k==5:
    print((ans5[r]-ans5[l-1])%(10**9+7))
```
优化后
```python
import sys

# 读取 n 和 m
n, m = map(int, input().split())

# 读取数组 a
a = list(map(int, input().split()))

# 预处理前缀和，ans[k][i] 记录 a[i]^k 的前缀和
K = 5  # 最大 k 的值
ans = [[0] * (n + 1) for _ in range(K + 1)]

# 计算 1~5 次方的前缀和
for i in range(1, n + 1):
    for k in range(1, K + 1):
        ans[k][i] = ans[k][i - 1] + a[i - 1] ** k  # 注意 a[i-1] 因为 a 是 0-based 索引

# 处理查询
mod = 10**9+7
for _ in range(m):
    l, r, k = map(int, input().split())
    print((ans[k][r] - ans[k][l - 1]) % mod )  # 计算区间 k 次方和并取模
```

- **bisect函数**

```python
bisect.bisect_left(lst, x)	返回 x 在 lst 中的插入位置（如果有相同元素，插在左侧）。
bisect.bisect_right(lst, x)	返回 x 在 lst 中的插入位置（如果有相同元素，插在右侧）。
bisect.insort_left(lst, x)	在 lst 中插入 x，保持有序（插入左侧）。
bisect.insort_right(lst, x)	在 lst 中插入 x，保持有序（插入右侧）。

bisect.bisect_left(arr, x, lo=0, hi=len(arr))

arr（序列）：这是你希望在其中查找插入位置的排序序列（例如一个升序排列的列表）。这个参数是必需的，表示我们要查找的目标列表。
x（目标值）：这是你想要插入的值，或者你希望查找它插入位置的值。bisect_left 会返回这个值在列表中的位置，确保插入后的列表仍然是有序的。例如，查找数字 x 应该插入到列表中的哪个位置，以保持列表的升序。
lo（可选，默认值为 0）：这是查找区间的起始位置。它是你希望开始查找的位置。如果不提供，默认从索引 0 开始查找。如果你只关心列表的一部分，可以通过指定 lo 来限制查找的范围。
hi（可选，默认值为 len(arr)）：这是查找区间的结束位置。它是你希望停止查找的位置，默认是列表的末尾。如果不提供，默认会查找整个列表。
```

- [M次方根](https://www.lanqiao.cn/problems/1542/learning/?page=1&first_category_id=1&tags=%E4%BA%8C%E5%88%86&tag_relation=intersection&sort=pass_rate&asc=0)
```python
import os
import sys
n, m = map(int, input().split())

# 设定左右边界
l, r = 1.0, n  
eps = 1e-8  # 增加精度，确保计算到 7 位小数

while r - l > eps:
    mid = (l + r) / 2
    power = mid**m

    if abs(power - n) < eps:  # 直接判断误差
        l = mid
        break
    elif power < n:
        l = mid
    else:
        r = mid

# 输出保留 7 位小数
print(f"{l:.7f}")
```
- [工厂质检员](https://www.lanqiao.cn/problems/8208/learning/?page=1&first_category_id=1&tags=%E4%BA%8C%E5%88%86&tag_relation=intersection&sort=pass_rate&asc=0)
```python
import sys

# 读取输入
n, k = map(int, input().split())
d_list = list(map(int, input().split()))

# 二分查找
l, r = 1, sum(d_list)  # r 取最大可能的高度
while l <= r:
    mid = (l + r) // 2  # 取整数中间值
    count = sum(d // mid for d in d_list)  # 计算能分出的块数
    
    if count >= k:
        l = mid + 1  # 可以分更多块，尝试增加高度
    else:
        r = mid - 1  # 块数不够，降低高度

# 如果 r < 1，说明无法分成 K 份
if r < 1:
    print(-1)
else:
    print(r)
```

- [区间覆盖增强版](https://www.luogu.com.cn/problem/P2082)

```python
n = int(input())
a = []

# 读取区间
for _ in range(n):
    l, r = map(int, input().split())
    a.append((l, r))

# 按起点排序
a.sort(key=lambda x: x[0])

merged = []
start, end = a[0]

for i in range(1, n):
    curr_start, curr_end = a[i]

    if curr_start <= end:  # 区间有重叠
        end = max(end, curr_end)
    else:
        # 不重叠，保存前一个合并区间
        merged.append((start, end))
        start, end = curr_start, curr_end

# 别忘了把最后一个区间加入
merged.append((start, end))

# 计算总长度
total = sum(r - l for l, r in merged)
print(total)
```
- **zip**
```python
list = sorted(map(int, input().split()))
least = min(b - a for a, b in zip(list, list[1:]))
```
**zip(list, list[1:])：**
把原列表和它的后移一位副本打包成一对对的形式。
例如 list = [2, 3, 5, 8]，那么：
```
zip(list, list[1:])  →  [(2, 3), (3, 5), (5, 8)]
b - a for a, b in zip(...)：
```
对每一对相邻的数 a 和 b，计算它们的差值 b - a。

min(...)：

取所有差值中的最小值。

对应上面的例子：
差值 = [1, 2, 3] → 最小的是 1，所以 least = 1

- [删除字符](https://www.lanqiao.cn/problems/544/learning/?page=1&first_category_id=1&tags=%E4%BA%8C%E5%88%86,%E4%BA%8C%E7%BB%B4%E5%89%8D%E7%BC%80%E5%92%8C,%E5%89%8D%E7%BC%80%E5%92%8C,%E8%B4%AA%E5%BF%83,%E5%B7%AE%E5%88%86&tag_relation=union&sort=students_count&asc=0)
```python
import os
import sys

# 请在此输入您的代码
s = input()
k = int(input())

stack = []

for c in s:
    while stack and k > 0 and stack[-1] > c:
        stack.pop()
        k -= 1
    stack.append(c)

# 如果还有没删完的，说明后面都字典序升序，可以直接去掉末尾的
while k > 0:
    stack.pop()
    k -= 1

print(''.join(stack))
```
- [求和](https://www.lanqiao.cn/problems/2080/learning/?page=1&first_category_id=1&tags=%E4%BA%8C%E7%BB%B4%E5%89%8D%E7%BC%80%E5%92%8C,%E5%89%8D%E7%BC%80%E5%92%8C&tag_relation=union&sort=students_count&asc=0)
```python
import os
import sys

# 请在此输入您的代码
n=int(input())
a=list(map(int,input().split()))
count=0
total=sum(a)
ans=0
for i in range(n):
  total-=a[i]
  ans+=a[i]*total
print(ans)
```
- **ord函数**
```python
print(ord('a') - ord('a'))  # 输出 0
print(ord('c') - ord('a'))  # 输出 2
```
- **round**函数(四舍五入)
```python
print(f'{round((youxiu/n)*100)}%')
```
- **字符串相加**
```python
print(chr(ord('a')+a[0][0]))
```

- **树状数组**

[树状数组1](https://www.luogu.com.cn/problem/P3374)
```python
n,m=map(int,input().split())
a=list(map(int,input().split()))

def lowbit(x):
    return x&-x

def query(x):
    ans=0
    while x>0:
        ans+=tr[x]
        x-=lowbit(x)
    return ans

def add(x,k):
    while x<=n:
        tr[x]+=k
        x+=lowbit(x)

tr=[0]*(n+1)

for i in range(1,n+1):
    add(i,a[i-1])

for _ in range(m):
    op=list(map(int,input().split()))
    if op[0]==1:
        add(op[1],op[2])
    if op[0]==2:
        print(query(op[2])-query(op[1]-1))
```
[树状数组2](https://www.luogu.com.cn/problem/P3368)
```python
n, m = map(int, input().split())
a = list(map(int, input().split()))

def lowbit(x):
    return x & -x

def query(x):
    res = 0
    while x > 0:
        res += tr[x]
        x -= lowbit(x)
    return res

def add(x, k):
    while x <= n:
        tr[x] += k
        x += lowbit(x)

tr = [0] * (n + 1)

# 正确初始化差分数组
for i in range(1, n + 1):
    delta = a[i - 1] - (a[i - 2] if i > 1 else 0)
    add(i, delta)

# 处理操作
for _ in range(m):
    op = list(map(int, input().split()))
    if op[0] == 1:
        x, y, k = op[1], op[2], op[3]
        add(x, k)
        if y + 1 <= n:
            add(y + 1, -k)
    else:
        x = op[1]
        print(query(x))
```

- 输入

```python
grid = [list(input().strip()) for _ in range(n)]
grid=[list(map(int,input().split())) for _ in range(n)]
```

- [小红的大蘑菇hard](https://ac.nowcoder.com/acm/contest/106504/C)

```python
from collections import deque

# 输入地图大小
n, m = map(int, input().split())
a = [[0] * (m + 4) for _ in range(n + 4)]  # 边界扩展
b = []

# 读取地图数据
for i in range(2, n + 2):
    row = input()
    for j in range(len(row)):
        if row[j] == '.':
            a[i][j + 2] = 0
            b.append((i, j + 2))
        elif row[j] == '*':
            a[i][j + 2] = 1
        else:
            a[i][j + 2] = 2

# 预处理可以通行的点
e = set()  # 用set加速查找
for c in b:
    if (a[c[0] - 1][c[1]] + a[c[0] + 1][c[1]] + a[c[0]][c[1] + 1] + a[c[0]][c[1] - 1]) == 0:
        # 检查周围是否有不安全的区域
        if a[c[0] - 2][c[1]] != 2 and a[c[0] - 1][c[1] - 1] != 2 and a[c[0] - 1][c[1] + 1] != 2 and \
           a[c[0]][c[1] - 2] != 2 and a[c[0] + 1][c[1] - 1] != 2 and a[c[0] + 2][c[1]] != 2 and \
           a[c[0] + 1][c[1] + 1] != 2 and a[c[0]][c[1] + 2] != 2:
            e.add((c[0], c[1]))  # 使用 set 保存有效点

# 更新地图状态
for i in range(2, n + 2):
    for j in range(2, m + 2):
        if (i, j) in e:
            a[i][j] = 1
        else:
            a[i][j] = 0

# BFS
f = True
q = deque([[2, 2, 0]])  # 用 deque 代替列表
dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]

while q:
    x, y, dis = q.popleft()  # deque 中的 pop操作是 O(1)
    if x == n + 1 and y == m + 1:
        f = False
        print(dis)
        break
    for xx, yy in dirs:
        tx = xx + x
        ty = yy + y
        if 2 <= tx < n + 2 and 2 <= ty < m + 2 and a[tx][ty] == 1:
            q.append([tx, ty, dis + 1])

if f:
    print(-1)
```

- [并查集](https://www.lanqiao.cn/problems/1135/learning/?problem_list_id=30&page=1)
```python
import os
import sys

# 请在此输入您的代码
# 并查集类
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # 每个学生的父节点初始化为自己
        self.rank = [1] * n  # 记录每个节点的秩，帮助优化合并操作

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:  # 如果不在同一个集合中
            # 按秩合并，较小的树连接到较大的树
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

# 主程序
n, m = map(int, input().split())  # n为学生数量，m为操作数量
uf = UnionFind(n)  # 初始化并查集

# 处理操作
for _ in range(m):
    op, x, y = map(int, input().split())
    if op == 1:  # 合并操作
        uf.union(x-1, y-1)  # 注意转换为从0开始的索引
    elif op == 2:  # 查询操作
        if uf.find(x-1) == uf.find(y-1):
            print("YES")
        else:
            print("NO")
```

- [Reduce函数]()

```python
from functools import reduce
reduce(function, iterable[, initializer])

numbers = [1, 2, 3, 4]
result = reduce(lambda x, y: x + y, numbers)
print(result)  # 输出: 10

numbers = [1, 2, 3, 4]
result = reduce(lambda x, y: x + y, numbers, 10)  # 初始值为 10
print(result)  # 输出: 20

numbers = [1, 2, 3, 4]
result = reduce(lambda x, y: x * y, numbers)
print(result)  # 输出: 24

numbers = [1, 2, 3, 4, 5]
# 先使用 filter 过滤出大于 2 的数，再用 reduce 求最大值
filtered = filter(lambda x: x > 2, numbers)
max_value = reduce(lambda x, y: x if x > y else y, filtered)
print(max_value)  # 输出: 5

numbers = [1, 2, 3, 4]
result = reduce(lambda x, y: x + (y * y), numbers)
print(result)  # 输出: 30 (1 + 4 + 9 + 16)

strings = ["Hello", " ", "world", "!"]
result = reduce(lambda x, y: x + y, strings)
print(result)  # 输出: Hello world!
```

