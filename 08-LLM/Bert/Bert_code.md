### **1. 环境准备**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 硬件配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
```

---

### **2. 核心模块实现**

#### **2.1 嵌入层（Embeddings）**
```python
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len=512, hidden_size=768):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)  # 词嵌入
        self.position_embed = nn.Embedding(max_len, hidden_size)  # 位置嵌入
        self.segment_embed = nn.Embedding(2, hidden_size)        # 段落嵌入
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        # 生成位置ID [0, 1, ..., seq_len-1]
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # 三部分嵌入相加
        token_emb = self.token_embed(input_ids)          # (batch, seq_len, hidden)
        position_emb = self.position_embed(position_ids) # (1, seq_len, hidden)
        segment_emb = self.segment_embed(segment_ids)    # (batch, seq_len, hidden)
        
        embeddings = token_emb + position_emb + segment_emb
        return self.dropout(self.layer_norm(embeddings))
```

---

#### **2.2 自注意力机制（Self-Attention）**
```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 定义 Q/K/V 的线性变换
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 生成 Q/K/V 并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # 处理注意力掩码（可选）
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax 归一化
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 加权聚合 Value
        attn_output = torch.matmul(attn_weights, V)
        
        # 拼接多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(attn_output)
```

---

#### **2.3 Transformer 编码器层（Encoder Layer）**
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, ff_dim=3072):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接
        attn_output = self.self_attn(self.norm1(x), attention_mask)
        x = x + attn_output
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x
```

---

#### **2.4 BERT 主干网络**
```python
class BERT(nn.Module):
    def __init__(self, vocab_size, num_layers=12, hidden_size=768, num_heads=12):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads) 
            for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, segment_ids, attention_mask=None):
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        return x
```

---

### **3. 预训练任务实现**

#### **3.1 Masked Language Model (MLM)**
```python
class BERTForMLM(nn.Module):
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = bert_model
        self.mlm_head = nn.Linear(bert_model.embeddings.token_embed.weight.size(1), vocab_size)
        self.mlm_head.weight = bert_model.embeddings.token_embed.weight  # 权重共享
        
    def forward(self, input_ids, segment_ids, attention_mask=None):
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)
        return self.mlm_head(hidden_states)
```

---

#### **3.2 Next Sentence Prediction (NSP)**
```python
class BERTForNSP(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.nsp_head = nn.Linear(bert_model.embeddings.token_embed.weight.size(1), 2)
        
    def forward(self, input_ids, segment_ids, attention_mask=None):
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)
        cls_output = hidden_states[:, 0, :]  # 取 [CLS] 位置的输出
        return self.nsp_head(cls_output)
```

---

### **4. 数据预处理与训练**

#### **4.1 生成模拟数据（示例）**
```python
# 词汇表大小（示例）
VOCAB_SIZE = 30522  # BERT-base 的词汇表大小

# 生成模拟输入
batch_size = 8
input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, 128)).to(device)  # 128 为序列长度
segment_ids = torch.cat([torch.zeros(4, 64), torch.ones(4, 64)], dim=1).long().to(device)
attention_mask = torch.ones(batch_size, 128).to(device)

# MLM 标签
mlm_labels = torch.randint(0, VOCAB_SIZE, (batch_size, 128)).to(device)
# NSP 标签（0/1）
nsp_labels = torch.randint(0, 2, (batch_size,)).to(device)
```

---

#### **4.2 训练循环（精简版）**
```python
# 初始化模型
bert = BERT(VOCAB_SIZE, num_layers=2)  # 为了显存限制，仅用2层
model_mlm = BERTForMLM(bert, VOCAB_SIZE).to(device)
model_nsp = BERTForNSP(bert).to(device)

# 定义优化器
optimizer = optim.AdamW([
    {'params': model_mlm.parameters(), 'lr': 2e-5},
    {'params': model_nsp.parameters(), 'lr': 2e-5}
])

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

# 训练循环
for epoch in range(3):
    model_mlm.train()
    model_nsp.train()
    
    # 前向传播（混合精度）
    with torch.cuda.amp.autocast():
        mlm_logits = model_mlm(input_ids, segment_ids)
        nsp_logits = model_nsp(input_ids, segment_ids)
        
        # 计算损失
        loss_mlm = nn.CrossEntropyLoss()(mlm_logits.view(-1, VOCAB_SIZE), mlm_labels.view(-1))
        loss_nsp = nn.CrossEntropyLoss()(nsp_logits, nsp_labels)
        total_loss = loss_mlm + loss_nsp
    
    # 反向传播
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")
```

---

### **5. 显存优化策略**
1. **梯度累积**：
   ```python
   accumulation_steps = 4  # 每4个批次更新一次参数
   for i, batch in enumerate(dataloader):
       with torch.cuda.amp.autocast():
           loss = model(batch)
           loss = loss / accumulation_steps  # 梯度缩放
       scaler.scale(loss).backward()
       
       if (i+1) % accumulation_steps == 0:
           scaler.step(optimizer)
           scaler.update()
           optimizer.zero_grad()
   ```

2. **动态序列长度**：
   ```python
   # 按批次中最长序列填充
   data_collator = lambda batch: {
       'input_ids': pad_sequence([x['input_ids'] for x in batch], batch_first=True),
       'segment_ids': pad_sequence([x['segment_ids'] for x in batch], batch_first=True),
       'attention_mask': pad_sequence([x['attention_mask'] for x in batch], batch_first=True)
   }
   ```

3. **模型简化**：
   ```python
   # 使用更小的模型配置
   bert = BERT(
       vocab_size=VOCAB_SIZE,
       num_layers=4,        # 减少层数
       hidden_size=512,     # 减小隐藏层维度
       num_heads=8
   )
   ```

---

### **6. 关键实现细节**
1. **权重共享**：MLM 头的权重与词嵌入层共享，减少参数量。
2. **注意力掩码**：通过 `attention_mask` 处理填充符（代码中未完整实现，需根据实际数据补充）。
3. **层归一化位置**：在残差连接之后进行归一化（Post-LN），与原始论文一致。
4. **激活函数**：使用 GELU 代替 RELU，与 BERT 原始设计一致。

---

### **7. 扩展功能建议**
1. **完整 MLM 遮盖策略**：实现 80% [MASK]、10% 随机替换、10% 保持原词。
2. **加载预训练权重**：从 Hugging Face 模型加载参数到自定义模型。
3. **支持长文本**：实现分段处理（如 512 Token 限制）。

