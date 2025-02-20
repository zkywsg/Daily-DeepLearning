#### **1. 核心思想**
多头注意力（Multi-Head Attention）通过并行运行多个独立的注意力头，使模型能够**同时关注输入序列的不同子空间信息**。每个头学习不同的特征表示，最后合并结果以增强模型的表达能力。

**2. 数学公式推导**

##### **2.1 输入定义**
- 输入矩阵： $X \in \mathbb{R}^{n \times d_{\text{model}}}$（n为序列长度， $d_{\text{model}}$为模型维度）
- 头数： $h$
- 每个头的维度： $d_k = d_v = d_{\text{model}} / h$

##### **2.2 线性投影**
对每个头进行独立的线性变换：

```math
\begin{aligned}
Q_i &= X W_i^Q \quad (W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}) \\
K_i &= X W_i^K \quad (W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}) \\
V_i &= X W_i^V \quad (W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v})
\end{aligned}
```



##### **2.3 缩放点积注意力**
每个头独立计算注意力：

```math
\text{head}_i = \text{softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i
```



##### **2.4 多头合并**
拼接所有头的输出并通过线性变换：

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O \quad (W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}})
```



---

### **3. 代码实现（PyTorch）**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)  # 整合所有头的Q
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        """将张量拆分为多个头 [batch_size, seq_len, d_model] → [batch_size, num_heads, seq_len, d_k]"""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
    def forward(self, q, k, v, mask=None):
        # 步骤1：线性投影并拆分多头
        q = self.split_heads(self.W_q(q))  # [batch, h, seq_len, d_k]
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        
        # 步骤2：计算缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 步骤3：加权求和
        output = torch.matmul(attn_weights, v)  # [batch, h, seq_len, d_k]
        
        # 步骤4：合并多头并线性变换
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)
```

---

### **4. 代码与公式对应解析**

| **公式步骤**                        | **代码实现**                                            |
| ----------------------------------- | ------------------------------------------------------- |
| 线性投影 $Q_i = X W_i^Q$            | `self.W_q(q)` 进行全头投影，`split_heads` 拆分为多头的Q |
| 计算 $\frac{Q_i K_i^T}{\sqrt{d_k}}$ | `torch.matmul(q, k.transpose(-2, -1)) / sqrt(d_k)`      |
| Softmax归一化                       | `F.softmax(attn_scores, dim=-1)`                        |
| 输出加权和 $head_i = A V_i$         | `torch.matmul(attn_weights, v)`                         |
| 合并多头并线性变换                  | `view` 合并多头维度，`self.W_o` 输出最终结果            |

---

### **5. 关键机制图解**

#### **5.1 多头拆分与合并流程**
```
输入张量形状：[batch_size, seq_len, d_model=512]
拆分多头后：[batch_size, num_heads=8, seq_len, d_k=64]
注意力计算后：[batch_size, 8, seq_len, 64]
合并多头后：[batch_size, seq_len, 512]
```

#### **5.2 多头注意力可视化**
![多头注意力示意图](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

---

### **6. 设计优势分析**

1. **并行化计算**：多个头可同时独立计算，充分利用GPU并行能力。
2. **多样化特征捕捉**：
   - 头1可能关注**局部语法结构**（如动词-宾语关系）
   - 头2可能捕捉**长距离指代**（如代词与其指代对象）
   - 头3可能学习**语义角色**（如施事者与受事者）
3. **模型容量扩展**：通过增加头数提升模型复杂度，但不显著增加计算量（因每个头的维度降低）。

---

### **7. 运行示例**

```python
# 参数设置
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

# 实例化模块
mha = MultiHeadAttention(d_model, num_heads)

# 模拟输入（假设q, k, v相同）
x = torch.randn(batch_size, seq_len, d_model)

# 前向计算
output = mha(x, x, x)
print(output.shape)  # torch.Size([2, 10, 512])
```

---

### **8. 常见问题解答**

**Q1: 为什么需要拆分到多个头，而不是直接用更大的单头？**  
- **答案**：多头允许模型在不同子空间独立学习特征，类似于CNN中多滤波器的设计理念。实验表明，8个头比单头大维度（如d_model=512）效果更好。

**Q2: 如何选择最佳头数？**  
- **经验法则**：通常设为$d_{\text{model}}$的约数（如512→8头，768→12头）。可通过消融实验调整。

**Q3: 不同头是否会学习到重复模式？**  
- **现象**：实际训练中，部分头会呈现相似行为，但整体呈现多样性。可通过可视化注意力权重观察（如使用[bertviz](https://github.com/jessevig/bertviz)）。

