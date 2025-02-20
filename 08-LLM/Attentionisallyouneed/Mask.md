#### **1. 核心目的**

- **防止信息泄漏**：确保解码器在生成当前位置的输出时，仅能访问已生成的先前位置信息，避免利用未来信息（即保证自回归特性）。
- **处理填充位置**：屏蔽无效的填充符号（Padding Tokens），防止模型关注无意义的输入部分。

**2. 掩码类型与作用**

##### **2.1 序列掩码（Sequence Mask）**
- **功能**：掩盖未来位置，强制模型仅关注当前位置及之前的词。
- **生成方式**：上三角矩阵（Upper Triangular Matrix），主对角线以上的位置设为掩码。
- **示例**（序列长度=4）：
  ```python
  [[0, -inf, -inf, -inf],
   [0, 0, -inf, -inf],
   [0, 0, 0, -inf],
   [0, 0, 0, 0]]  # 0表示保留，-inf表示掩盖
  ```

##### **2.2 填充掩码（Padding Mask）**
- **功能**：屏蔽输入中的填充符号（如`<pad>`）。
- **生成方式**：标记所有填充符号的位置为掩码。
- **示例**（有效长度=3，填充后长度=5）：
  ```python
  [False, False, False, True, True]  # True表示需要掩盖的位置
  ```

##### **2.3 合并掩码**
- **逻辑操作**：`合并掩码 = 序列掩码 | 填充掩码`
- **作用**：同时屏蔽未来位置和填充位置。

---

### **3. 数学原理**
#### **3.1 注意力分数计算**
原始注意力分数：

```math
\text{Attention Scores} = \frac{QK^T}{\sqrt{d_k}}
```



#### **3.2 掩码应用**
将掩码矩阵 $M$（掩码位置为`-inf`）与注意力分数相加：

```math
\text{Masked Scores} = \text{Attention Scores} + M
```



#### **3.3 Softmax归一化**
经过掩码后的Softmax计算：

```math
\text{Attention Weights} = \text{softmax}(\text{Masked Scores})
```



**效果**：被掩码的位置权重趋近于0，模型仅关注有效位置。

---

### **4. 代码实现（PyTorch）**
#### **4.1 生成序列掩码**
```python
def generate_square_subsequent_mask(sz):
    """生成上三角掩码矩阵"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1  # 上三角为True
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask  # 形状: [sz, sz]
```

#### **4.2 生成填充掩码**
```python
def create_pad_mask(seq, pad_token=0):
    """标记填充符号位置"""
    return (seq == pad_token)  # 形状: [batch_size, seq_len]
```

#### **4.3 合并掩码并应用**
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # ...其他层定义

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        # 自注意力（应用序列掩码+填充掩码）
        attn_output = self.self_attn(
            x, x, x, 
            mask=tgt_mask  # 合并后的掩码
        )
        # 交叉注意力（仅用源序列掩码）
        cross_output = self.cross_attn(
            attn_output, encoder_out, encoder_out,
            mask=src_mask
        )
        # ...后续处理
        return output
```

#### **4.4 掩码应用流程**
```python
# 假设输入目标序列形状: [batch_size, tgt_seq_len]
tgt_seq = ...  # 右移后的目标序列
pad_mask = create_pad_mask(tgt_seq)  # [batch_size, tgt_seq_len]
seq_mask = generate_square_subsequent_mask(tgt_seq.size(1))  # [tgt_seq_len, tgt_seq_len]

# 合并掩码（广播机制）
combined_mask = pad_mask.unsqueeze(1) | seq_mask.unsqueeze(0)  # [batch_size, tgt_seq_len, tgt_seq_len]

# 前向传播时传入合并掩码
decoder_output = decoder_layer(x, encoder_out, src_mask, combined_mask)
```

---

### **5. 关键机制图解**
#### **5.1 掩码叠加效果**
```
原始序列: ["A", "B", "<pad>", "<pad>"]
有效位置: [  1,    1,      0,      0   ]

序列掩码（未来位置）     填充掩码         合并掩码
[0 -inf -inf -inf]    [0 0 1 1]    [0 -inf -inf -inf]
[0  0  -inf -inf]     [0 0 1 1] →   [0  0  -inf -inf]
[0  0   0  -inf]      [1 1 1 1]     [1  1   1  -inf]
[0  0   0   0 ]       [1 1 1 1]      [1  1   1   1 ]
```

#### **5.2 注意力权重可视化**
![掩码注意力权重](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

---

### **6. 应用场景示例**
**任务**: 英译中 `"I love NLP" → "我 爱 自然语言处理"`

**生成步骤**：
1. 输入 `<start>` → 输出 "我"（仅能关注起始符）
2. 输入 `<start> 我` → 输出 "爱"（关注起始符和"我"）
3. 输入 `<start> 我 爱` → 输出 "自然语言处理"（关注所有已生成词）

**掩码作用**：确保生成"爱"时无法看到后续的"自然语言处理"。

---

### **7. 常见问题解答**
**Q1: 为什么用上三角矩阵而非下三角？**  
- 上三角矩阵更直观表示"未来位置"，且与矩阵乘法维度对齐。

**Q2: 如何处理变长序列的掩码？**  
- 动态生成与当前序列长度匹配的掩码矩阵。

**Q3: 推理时如何应用掩码？**  
- 自回归生成时逐步扩展掩码，每次生成新词后更新掩码。
