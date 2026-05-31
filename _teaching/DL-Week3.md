---
title: "第3周：Transformer模型"
collection: teaching
permalink: /teaching/DL-Week3
---

<img src="https://raw.githubusercontent.com/HITWang-tech/HITWang-tech.github.io/refs/heads/master/images/exported_image.png">

## 1. 诞生与核心思想
### 1.1 背景
- 之前序列模型以 RNN（LSTM, GRU）和 CNN（ConvS2S）为主。
- RNN 的缺陷：难以并行、长距离依赖易梯度消失、训练慢。
- CNN 的缺陷：需要多层才能扩大感受野，对全局依赖建模不直接。

### 1.2 论文与核心贡献
- **论文**：《Attention Is All You Need》（Vaswani et al., 2017）
- **核心思想**：只用注意力机制，完全抛弃循环与卷积。
- **两大优势**：
  1. **长距离依赖**：任意两个位置的直接连接，路径长度 O(1)。
  2. **并行计算**：所有位置可同时计算，适合 GPU。

### 1.3 整体架构
- 经典编码器-解码器结构（机器翻译任务）。
- 编码器：6 层堆叠，每层包含「多头自注意力 + FFN」。
- 解码器：6 层堆叠，每层包含「掩码多头自注意力 + 交叉注意力 + FFN」。
- 输入输出均加位置编码。
  
---

## 2. 输入部分
### 2.1 Token 嵌入
- `nn.Embedding(vocab, d_model)`
  
### 2.2 位置编码
- **为什么需要**：自注意力置换不变，无时序概念
- **正弦/余弦编码**（原始）
  - 公式： $$PE(pos,2i)=sin(pos/10000^(2i/d))，PE(pos,2i+1)=cos(...)$$
  - 特点：确定、可外推、多频率
- **其他方案**：可学习、相对位置、RoPE（大模型主流）、ALiBi
  
### 2.3 输出嵌入
- 目标语言词典，结构相同
  
---

## 3. 注意力机制
### 3.1 缩放点积注意力


 #### 3.1.1 公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

- 除以 $$√d_k$$ ：防 softmax 饱和、稳梯度
- $$Q$$ (Query) : 当前单词的查询向量，形状 `(seq_len_q, d_k)`
- $$K$$ (Key)   : 所有单词的键向量，形状 `(seq_len_k, d_k)`
- $$V$$ (Value) : 所有单词的值向量，形状 `(seq_len_v, d_v)`（通常 $$d_v = d_k$$）

#### 3.1.2 计算步骤
1. 计算 $$QK^T$$ ，得到注意力分数矩阵（每对位置的相似度）。
2. 除以 $$\sqrt{d_k}$$ 进行缩放。
3. 可选：应用掩码（Padding Mask / Look-ahead Mask）。
4. softmax 归一化，得到权重矩阵。
5. 加权求和：权重矩阵 × $$V$$ 。

#### 3.1.3 为什么要除以 $$\sqrt{d_k}$$ ？
- 当 $$d_k$$ 较大时，点积的绝对值容易变得很大，导致 softmax 进入饱和区（梯度极小）。
- 除以 $$\sqrt{d_k}$$ 使点积的方差保持在 1，稳定梯度。

### 3.2 多头注意力（Multi-Head Attention）

#### 3.2.1 动机
- 单头注意力只能学习一种相关性模式。
- 多头可以从不同子空间、不同角度捕捉多种依赖关系（如语法、语义、长距离等）。

#### 3.2.2 计算过程
1. 对输入 $$X$$（shape: `[batch, seq_len, d_model]`）线性投影得到 $$Q, K, V$$：

   $$
   Q_i = X W_i^Q,\quad K_i = X W_i^K,\quad V_i = X W_i^V
   $$
   
2. 将 $$Q_i, K_i, V_i$$ 按头数切分（`d_model / h` 维度）。
3. 每个头独立执行缩放点积注意力。
4. 将所有头的输出拼接，再通过一个输出线性变换：
   
   $$
   \text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
   $$

#### 3.2.3 维度变换
- 输入：`(B, S, D)`，其中 $$D = d_{\text{model}}$$
- 分头：`(B, S, H, D/H)` → 转置为 `(B, H, S, D/H)`
- 每个头计算后：`(B, H, S, D/H)`
- 拼接：`(B, S, D)`
- 输出线性：`(B, S, D)`

### 3.3 自注意力（Self-Attention）与交叉注意力（Cross-Attention）
- **自注意力**：$$Q, K, V$$ 来自同一序列。编码器中使用，双向（能看到整个句子）。
- **交叉注意力**：$$Q$$ 来自解码器上一层的输出，$$K, V$$ 来自编码器的输出。解码器中间层使用，让解码器关注输入序列的相关部分。

### 3.4 掩码（Mask）
- **Padding Mask**：对 batch 中填充的 `<PAD>` 位置，将其注意力分数置为 $$-\infty$$，使 softmax 后权重为 0。
- **Look-ahead Mask（因果掩码）**：在解码器自注意力中，位置 $$i$$ 只能看到 $$j \le i$$ 的位置。上三角部分设为 $$-\infty$$。
- **组合**：解码器自注意力同时使用 Padding Mask + Look-ahead Mask。

### 3.5 代码示例（单头注意力）
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, V), p_attn
```

### 3.2 多头注意力（Multi-Head Attention）

#### 3.2.1 动机
- 单头注意力只能学习一种相关性模式。
- 多头可以从不同子空间、不同角度捕捉多种依赖关系（如语法、语义、长距离等）。

#### 3.2.2 计算过程
1. 对输入 $$X$$（shape: `[batch, seq_len, d_model]`）线性投影得到 $$Q, K, V$$：

   $$
   Q_i = X W_i^Q,\quad K_i = X W_i^K,\quad V_i = X W_i^V
   $$
   
2. 将 $$Q_i, K_i, V_i$$ 按头数切分（`d_model / h` 维度）。
3. 每个头独立执行缩放点积注意力。
4. 将所有头的输出拼接，再通过一个输出线性变换：
   
   $$
   \text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
   $$

#### 3.2.3 维度变换
- 输入：`(B, S, D)`，其中 $$D = d_{\text{model}}$$
- 分头：`(B, S, H, D/H)` → 转置为 `(B, H, S, D/H)`
- 每个头计算后：`(B, H, S, D/H)`
- 拼接：`(B, S, D)`
- 输出线性：`(B, S, D)`

### 3.3 自注意力（Self-Attention）与交叉注意力（Cross-Attention）
- **自注意力**：$$Q, K, V$$ 来自同一序列。编码器中使用，双向（能看到整个句子）。
- **交叉注意力**：$$Q$$ 来自解码器上一层的输出，$$K, V$$ 来自编码器的输出。解码器中间层使用，让解码器关注输入序列的相关部分。

### 3.4 掩码（Mask）
- **Padding Mask**：对 batch 中填充的 `<PAD>` 位置，将其注意力分数置为 $$-\infty$$，使 softmax 后权重为 0。
- **Look-ahead Mask（因果掩码）**：在解码器自注意力中，位置 $$i$$ 只能看到 $$j \le i$$ 的位置。上三角部分设为 $$-\infty$$。
- **组合**：解码器自注意力同时使用 Padding Mask + Look-ahead Mask。

### 3.5 代码示例（单头注意力）
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, V), p_attn
```

--- 
## 4 编码器 Encoder

### 4.1 编码器层结构
每个编码器层包含两个子层：
1. **多头自注意力子层**（+ 残差 + LayerNorm）
2. **前馈网络子层**（+ 残差 + LayerNorm）

### 4.2 残差连接（Residual Connection）
- 公式：`output = LayerNorm(x + Sublayer(x))`  （Post-LN 原始写法）
- 现代常用 Pre-LN：`output = x + Sublayer(LayerNorm(x))`，训练更稳定。
- 作用：缓解梯度消失，使深层网络可训练。

### 4.3 层归一化（Layer Normalization）
- 对每个样本的 **特征维度** 进行归一化（均值为 0，方差为 1），再学习缩放和平移参数。
- 与 BatchNorm 对比：
  
  |         | LayerNorm | BatchNorm |
  |---------|-----------|------------|
  | 归一化维度 | 特征维（C/H） | 批维（N） |
  | 适合序列 | ✅ 变长 | ❌ 依赖 batch 统计 |
  | 训练/测试一致 | ✅ | ❌ 需跟踪移动平均 |
  
- Transformer 中全部使用 LayerNorm。

### 4.4 前馈网络（FFN）
- 结构：两个线性层，中间有激活函数（ReLU 或 GELU）。
- 公式：
  $$
  \text{FFN}(x) = \text{ReLU}(xW_1 + b_1) W_2 + b_2
  $$
- 参数：通常 `d_ff = 4 * d_model`，即先升维再降维。
- 作用：对每个位置做独立非线性变换，与注意力的全局混合互补。
- 现代升级：使用 SwiGLU（LLaMA, PaLM），参数更多，效果更好。

### 4.5 堆叠 N 层
- 原始论文 N=6，每一层的输出维度保持 `(batch, seq_len, d_model)`。
- 浅层学习局部/短语特征，深层学习全局/语义特征。

---

## 5 解码器 Decoder

### 5.1 解码器层结构
每个解码器层包含三个子层：
1. **带掩码的多头自注意力**（输出 `(B,S,D)`）
2. **交叉注意力**（Q 来自子层1，K、V 来自编码器输出）
3. **前馈网络**

每个子层后都有残差连接 + LayerNorm（Pre-LN 或 Post-LN）。

### 5.2 掩码自注意力
- 因果掩码（上三角矩阵）确保自回归性质。
- 训练时 teacher forcing 也需要掩码，防止作弊。

### 5.3 交叉注意力
- Q: 解码器当前状态（已生成的序列）
- K, V: 编码器输出的整个源序列的表示
- 作用：让解码器在生成每一步时，动态选择编码器中最相关的信息。

### 5.4 初始输入
- 解码器第一个输入是 `<SOS>` 标记（start of sentence），然后逐词预测。

---
## 6 输出层与损失函数

### 6.1 输出线性层 + Softmax
- 线性层：将解码器输出 `(batch, seq_len, d_model)` 映射到 `(batch, seq_len, vocab_size)`。
- Softmax：沿 vocab 维度归一化，得到每个位置的概率分布。

### 6.2 损失函数
- 训练时：交叉熵损失，忽略 padding 位置的 loss。
- 公式：
  $$
  \mathcal{L} = -\sum_{t=1}^{T} \log p(y_t | y_{<t}, x)
  $$

---

## 7 完整模型前向计算

### 7.1 训练模式（以机器翻译为例）
1. 源句子 `src` → 词嵌入 + 位置编码 → 编码器输出 `memory`。
2. 目标句子 `tgt`（shifted right，即前面加 `<SOS>`，去掉最后一个 token）→ 词嵌入 + 位置编码 → 解码器。
3. 解码器每层：掩码自注意力 → 交叉注意力（使用 `memory`）→ FFN。
4. 解码器最终输出通过线性+softmax，预测每个位置的词。
5. 计算交叉熵损失（与真实 `tgt` 比较，通常真实 tgt 是去掉 `<SOS>` 的原始目标序列）。

### 7.2 推理模式（自回归生成）
1. 编码源句子一次，得到 `memory`。
2. 解码器初始输入为 `<SOS>`，预测第一个词。
3. 将新词拼接到输入，重复直到生成 `<EOS>` 或达到最大长度。
4. 可选 Beam Search 等策略。

---

## 8 训练细节与优化技巧

### 8.1 数据预处理
- **BPE（Byte Pair Encoding）**：子词分词，平衡词表大小与 OOV。
- **序列对齐**：将 batch 内的句子按长度排序，并 padding 到相同长度。
- **Mask 构造**：Padding mask 根据 token 是否为 `<PAD>` 产生。

### 8.2 标签平滑（Label Smoothing）
- 将 one‑hot 目标分布替换为：
  $$
  y' = (1 - \epsilon) y_{\text{one-hot}} + \epsilon / \text{vocab\_size}
  $$
- 作用：防止模型过于自信，提升泛化能力。
- 原始 Transformer 使用 $$\epsilon=0.1$$。

### 8.3 学习率调度（Warmup）
- 先线性增加到最大值，再按平方根倒数衰减。
- 公式：
  $$
  \text{lr} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})
  $$
- 原因：早期梯度不稳定，小 lr 防止 divergence；后期逐步衰减。

### 8.4 参数初始化
- 所有线性层权重：均值为 0，方差为 $$1/\sqrt{d_{\text{model}}}$$ 的正态分布。
- 嵌入层：方差为 1。
- 编码器/解码器最后的线性层额外乘系数 $$1/\sqrt{2N}$$（根据残差堆叠深度）。

---

## 9 推理与解码策略

### 9.1 贪心搜索
- 每一步选择概率最大的 token。
- 缺点：局部最优，可能错过全局更好的序列。

### 9.2 束搜索（Beam Search）
- 每一步保留 top-k 个候选序列（k 为束宽）。
- 最终选择对数概率最高的完整序列。
- 通常束宽设为 4～8。
- 长度归一化：除以 `(len)^α` 惩罚短句倾向。

### 9.3 KV Cache
- 自回归生成时，每步都要用之前所有 token 的 K、V 计算注意力。
- 将之前步骤的 K、V 缓存起来，新 token 只计算自己的 K、V，然后与缓存拼接。
- 极大减少重复计算，是推理加速的关键技术。

---

## 10 Transformer 变体与进化

### 10.1 Decoder-only（生成式，GPT 系列）
- 结构：仅解码器 + 因果掩码。
- 训练任务：自回归语言模型（预测下一个 token）。
- 代表：GPT-1/2/3/4, LLaMA, Qwen, Mistral, DeepSeek。
- 特点：大规模预训练 + 指令微调，适用于对话、代码、推理。

### 10.2 Encoder-only（理解式，BERT 系列）
- 结构：仅编码器，双向注意力。
- 训练任务：MLM（掩码语言模型）+ NSP（下一句预测，后来常去掉）。
- 代表：BERT, RoBERTa, ALBERT, DeBERTa。
- 适用：文本分类、命名实体识别、句子相似度。

### 10.3 Encoder-Decoder（文本到文本）
- 代表：T5, BART, PEGASUS。
- 统一框架：将所有 NLP 任务视为 text-to-text。
- 优势：适合翻译、摘要、改写等需要输入到输出的任务。

### 10.4 高效 Transformer（长序列优化）
- **稀疏注意力**：Longformer, BigBird，用滑动窗口 + 全局注意力降低 $$O(n^2)$$。
- **线性注意力**：Performer, Linear Transformer，用核方法逼近 softmax，复杂度 $$O(n)$$。
- **分块局部注意力**：Swin Transformer（视觉），将图像划分为窗口。
- **FlashAttention**：IO 感知的精确注意力，通过分块减少 GPU 内存读写，速度提升 2-4 倍。
- **多查询注意力 MQA**：所有头共享同一个 K、V，减少 KV Cache 大小。
- **分组查询注意力 GQA**：介于 MQA 和多头之间，LLaMA 2/3 使用。

### 10.5 位置编码进阶
- **RoPE（Rotary Position Embedding）**：将绝对位置以旋转矩阵形式乘到 Q、K 上，等价于相对位置编码，支持长外推。目前最流行。
- **ALiBi**：不嵌入位置，而是在注意力分数上减去一个与距离成正比的偏置。简单有效。

### 10.6 视觉/多模态 Transformer
- **ViT**：将图像切成 patch，作为 token 序列输入标准 Transformer。
- **CLIP**：双塔结构（图像编码器 + 文本编码器），对比学习图文对齐。
- **跨模态注意力**：用于多模态生成（如 Flamingo, BLIP）。

---

## 11 复杂度分析与长序列挑战

| 模型 | 每层时间复杂度 | 长序列瓶颈 |
|------|--------------|------------|
| RNN  | $$O(n \cdot d^2)$$ | 串行计算，无法并行 |
| CNN  | $$O(n \cdot k \cdot d^2)$$ | 需要 $$O(n/k)$$ 层才能看到全局 |
| Transformer 注意力 | $$O(n^2 \cdot d)$$ | 序列长度 n 增大时平方增长 |

- 当 n=1000 时，注意力矩阵 1M 元素；n=100k 时 10B 元素，不可行。
- 解决方案：稀疏注意力、线性注意力、混合架构（如 Mamba）。

---

## 12 常见问题与深入思考

1. **为什么 Transformer 需要位置编码？**
   - 自注意力是置换不变的，没有时序信息。
1. **为什么缩放点积要除以 $$\sqrt{d_k}$$？**
   - 防止点积过大导致 softmax 饱和，梯度消失。
3. **多头注意力为什么有效？**
   - 每个头关注不同的特征子空间，捕捉多种依赖类型。
4. **LayerNorm 放在残差前还是后？**
   - Post-LN（原始）：收敛不稳定但最终精度略好；Pre-LN（现代）：训练稳定，梯度流更顺畅。
5. **Decoder 的交叉注意力为什么 Q 来自解码器，K、V 来自编码器？**
   - 让解码器查询源序列中的相关信息，相当于注意力机制的翻译对齐。
6. **Transformer 如何做到并行训练？**
   - 训练时所有位置同时计算，无需像 RNN 那样等待前一步。
7. **Transformer 相对于 RNN 的劣势？**
   - 长序列显存 O(n²) 过大；缺乏循环结构导致的“无限记忆”能力。
8. **FlashAttention 的核心思想？**
   - 将 Q、K、V 分块，在 GPU 的 SRAM 中计算，避免频繁读写 HBM，同时保证数值精确。
9. **RoPE 如何实现相对位置编码？**
   - 将 Q、K 的向量按维度两两旋转，旋转角度与位置差线性相关，点积结果自然包含相对位置信息。
10. **预训练语言模型为什么 decoder-only 成为主流？**
    - 生成任务统一性好，可用自回归 loss 大规模训练，且工程上简洁（无需编码器）。

---

## 13 从零实现最小化 Transformer 的伪代码

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_decoder_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        # src, tgt: (batch, seq_len)
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask, src_padding_mask)
        output = tgt_emb
        for layer in self.decoder:
            output = layer(output, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)
        return self.fc_out(output)
```
实际实现需要处理 mask 维度、残差、LayerNorm 顺序等细节。









