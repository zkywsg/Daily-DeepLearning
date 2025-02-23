### 一、Self-Attention核心原理
#### 1. 核心思想
Self-Attention允许序列中的每个位置直接关注所有其他位置，通过动态计算注意力权重来捕捉长距离依赖关系。

#### 2. 计算流程图示
```
输入向量 → 线性变换 → Q,K,V → 注意力分数 → Softmax → 加权求和 → 输出
```

---

### 二、数学公式推导（分步骤解析）

#### 步骤1：输入表示
假设输入序列有n个词，每个词向量维度为 $d_{model}$：

```math
X \in \mathbb{R}^{n\times d_{model}}
```



#### 步骤2：生成Q/K/V
通过可学习的权重矩阵进行线性变换：

```math
\begin{aligned}
Q &= XW^Q \quad (W^Q \in \mathbb{R}^{d_{model} \times d_k}) \\
K &= XW^K \quad (W^K \in \mathbb{R}^{d_{model} \times d_k}) \\
V &= XW^V \quad (W^V \in \mathbb{R}^{d_{model} \times d_v})
\end{aligned}
```



#### 步骤3：计算注意力分数
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**分步推导**：
1. 相似度计算：  
   $S = QK^T \in \mathbb{R}^{n \times n}$
   
2. 缩放操作：  
   $S_{scaled} = \frac{S}{\sqrt{d_k}}$
   （防止点积值过大导致softmax梯度消失）

3. 归一化：  
   $A = \text{softmax}(S_{scaled}) \in \mathbb{R}^{n \times n}$

4. 加权求和：  
   $\text{Output} = AV \in \mathbb{R}^{n \times d_v}$

---

### 三、完整代码实现（带数学公式注释）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_k)  # Q变换矩阵
        self.WK = nn.Linear(d_model, d_k)  # K变换矩阵
        self.WV = nn.Linear(d_model, d_v)  # V变换矩阵
        self.scale = d_k ** 0.5  # 缩放因子√d_k

    def forward(self, x, mask=None):
        """
        x: 输入张量 [batch_size, seq_len, d_model]
        mask: 可选掩码 [batch_size, seq_len, seq_len]
        返回: [batch_size, seq_len, d_v]
        """
        # Step 1: 生成Q/K/V
        Q = self.WQ(x)  # [batch, seq, d_k]
        K = self.WK(x)  # [batch, seq, d_k]
        V = self.WV(x)  # [batch, seq, d_v]

        # Step 2: 计算QK^T（点积相似度）
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq, seq]

        # Step 3: 缩放操作
        attn_scores = attn_scores / self.scale  # 对应公式中的除以√d_k

        # Step 4: 应用掩码（如需要）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Step 5: Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq, seq]

        # Step 6: 加权求和
        output = torch.matmul(attn_weights, V)  # [batch, seq, d_v]
        return output
```

---

### 四、代码与公式对应关系图解
```
公式步骤         代码实现
---------------------------------------------------------
Q = XW^Q       → self.WQ(x)
QK^T          → torch.matmul(Q, K.transpose(-2, -1))
除以√d_k       → attn_scores / self.scale
softmax       → F.softmax()
AV            → torch.matmul(attn_weights, V)
```

---

### 五、关键机制解析
#### 1. 为什么要用三个不同的矩阵（Q/K/V）？
- **Q（Query）**：当前关注的位置
- **K（Key）**：被比较的位置
- **V（Value）**：实际提供信息的表示

#### 2. 缩放因子为什么是√d_k？
- 当维度 $d_k$较大时，点积结果会趋于极大值，导致softmax梯度消失
- 数学证明：假设 $q_i$和 $k_i$是独立随机变量，均值为0，方差为1，则 $q \cdot k$的方差为 $d_k$

#### 3. Softmax的作用
- 将注意力分数转换为概率分布
- 公式示例：  
  假设三个位置的分数为[2.0, 1.0, 0.1]，经过softmax后变为[0.65, 0.24, 0.11]

---

### 六、运行示例
```python
# 参数设置
d_model = 512
d_k = 64
d_v = 64
seq_len = 10
batch_size = 2

# 实例化模块
sa = SelfAttention(d_model, d_k, d_v)

# 模拟输入
x = torch.randn(batch_size, seq_len, d_model)

# 前向计算
output = sa(x)
print(output.shape)  # torch.Size([2, 10, 64])
```

---

### 七、多头注意力的扩展
将单头注意力扩展为h个头：
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.heads = nn.ModuleList([
            SelfAttention(d_model, self.d_k, self.d_k)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * self.d_k, d_model)

    def forward(self, x):
        return self.linear(
            torch.cat([h(x) for h in self.heads], dim=-1)
        )
```

---

### 八、应用场景示例（机器翻译）
```text
输入句子："The animal didn't cross the street because it was too tired"

当处理"it"时，自注意力权重可能显示：
it → animal (0.6)
it → street (0.3)
其他词 → (0.1)
```

通过这样的权重分配，模型能正确建立"it"与"animal"的指代关系。