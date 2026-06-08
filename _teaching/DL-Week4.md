## 1 大模型技术概述

### 1.1 大模型的定义与特点
- **定义**：指参数规模巨大（通常>10亿）、在大规模无标注数据上训练，并能通过微调或提示词适配多种任务的深度学习模型。
- **核心特点**：
  - **涌现能力**：参数量超过阈值后，模型表现出小模型不具备的推理、上下文学习等高级能力。
  - **统一架构**：以Transformer为基础，统一处理文本、图像、语音等模态。
  - **预训练+微调范式**：先在海量数据上自监督预训练，再在少量下游任务数据上适配。

### 1.2 发展历程
- **统计语言模型（N-gram）** → **神经语言模型（RNN/LSTM）** → **预训练模型（ELMo、BERT、GPT-1/2）** → **大模型时代（GPT-3、PaLM、LLaMA、GPT-4）**。
- 关键转折点：**Transformer架构**（2017）、**Scaling Law**（2020）、**指令微调与RLHF**（2022）。

### 1.3 核心技术要素
- **模型架构**：几乎全采用Transformer Decoder-only（GPT系列）或Encoder-Decoder（T5）。
- **训练数据**：万亿token级别，来源于网页、书籍、代码、科学论文等。
- **分布式训练**：3D并行（数据并行、流水并行、张量并行）、ZeRO优化、混合精度训练。
- **微调与对齐**：
  - **指令微调**：通过(指令, 输出)数据提升遵循指令能力。
  - **RLHF**：利用人类反馈强化学习对齐人类偏好。

### 1.4 大模型的影响
- 推动AI从“单任务专用”走向“通用能力底座”。
- 催生**AI Agent**、**多模态大模型**、**具身智能**等新方向。
- 带来计算成本、数据版权、可解释性、安全性等新挑战。
以下是扩展后的完整笔记，在原有六节基础上增加了 **代码示例**、**损失函数推导** 与 **训练超参数表**。新增内容以 📌 标记，方便查阅。

## 2 多模态大模型技术

### 2.1 多模态概念
- **模态**：数据的不同来源或形式，如文本、图像、音频、视频、3D点云。
- **多模态学习**：同时处理和理解多种模态信息，实现跨模态理解、生成与对齐。

### 2.2 关键挑战
- **异构鸿沟**：不同模态数据结构、分布不同（文本离散，图像连续）。
- **对齐问题**：需要在语义空间对齐时间与空间上对应的多模态信息。
- **融合效率**：如何高效融合长序列多模态特征。

### 2.3 主流多模态大模型架构
- **架构类型**：
  - **单编码器双解码器**：如Flamingo，一个视觉编码器 + 语言模型，通过交叉注意力融合。
  - **统一Transformer**：如Unified-IO、CM3leon，所有模态视为token序列。
  - **模块化组合**：如BLIP-2、LLaVA，用Q-Former或MLP连接预训练的视觉编码器与LLM。

- **代表性模型**：
  - **CLIP**（对比图文预训练）→ 多模态理解基础。
  - **Flamingo**：支持图像/视频输入，通过门控交叉注意力插入LLM层。
  - **GPT-4V**、**Gemini**：闭源商业多模态大模型，支持图像推理。
  - **LLaVA**：开源的视觉指令微调模型，将图像特征投影到LLM输入空间。

### 2.4 训练策略
- **阶段一：对齐预训练**：用图文对进行对比学习（CLIP风格）或生成式预训练。
- **阶段二：多模态指令微调**：构建包含图像/视频的对话数据，微调LLM使其遵循多模态指令。

### 2.5 评估与应用
- **评估基准**：MMMU（大学水平多模态推理）、SEED-Bench、MM-Vet。
- **应用**：视觉问答（VQA）、图像描述、交互式Agent、机器人操作、视频理解。

---

## 3 视觉Transformer（ViT）

### 3.1 从CNN到ViT
- **CNN局限**：局部感受野，难以建模全局依赖；需要多层堆叠才能扩大视野。
- **ViT核心思想**：将图像视为**序列**（patch序列），直接使用Transformer处理，天然具备全局感受野。

### 3.2 ViT模型结构
- **图像分块（Patchify）**：将H×W×C图像切为N个p×p×C的patch，N=H*W/p²。
- **线性投影（Patch Embedding）**：每个patch展平后线性映射到D维，得到N个token。
- **位置编码**：加可学习的1D位置编码（或相对位置编码、2D正弦编码）。
- **类别令牌（可选）**：额外加一个可学习的[CLS] token，用于分类任务。
- **Transformer编码器**：L层Multi-Head Self-Attention + FFN + LayerNorm + 残差连接。
- **分类头**：取[CLS] token的输出或全局平均池化，接MLP分类。
### 3.3 代码示例：ViT 完整前向实现（PyTorch）
```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, embed_dim, H', W') -> (B, N, D)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                     # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)       # (B, N+1, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]                           # 取[CLS] token
        return self.head(cls_out)
```

### 3.4 损失函数推导：ViT 分类任务
使用标准交叉熵损失：
$$
\mathcal{L} = -\sum_{c=1}^{C} y_c \log \hat{y}_c
$$
其中 \$$hat{y} = \text{softmax}$$text{head}(z_{\text{cls}}))$$。

### 3.5 训练超参数表（ViT‑Base/16 在 ImageNet‑1K）

| 超参数             | 典型值                          |
| ------------------ | ------------------------------- |
| 图像尺寸           | 224×224                         |
| Patch尺寸          | 16×16                           |
| 优化器             | AdamW (lr=3e-3, weight decay=0.3) |
| 学习率调度         | Cosine decay, warmup 10k steps  |
| 批量大小           | 4096 (使用1024 TPU cores)       |
| 训练轮数           | 300 epochs                      |
| 数据增强           | RandAugment (2, 15), Mixup (0.2), Cutmix (1.0) |
| 随机深度率         | 0.1                             |

### 3.6 关键改进与变体
- **DeiT**：引入知识蒸馏（教师CNN指导学生ViT），解决ViT需海量数据问题。
- **Swin Transformer**：引入移动窗口注意力，将计算复杂度从O(N²)降为O(N)。
- **PVT**：金字塔结构，逐渐降低空间分辨率，适合作为密集预测任务的骨干。
- **ViT适配器**：Adapter、LoRA等参数高效微调方法用于ViT下游任务。

### 3.7 训练与特性
- **训练数据需求**：原始ViT需JFT-300M（3亿图）才能超越CNN；DeiT用ImageNet可达到类似效果。
- **归纳偏置削弱**：ViT没有CNN的平移等变性和局部性，全靠数据学习，因此需要更大数据或正则化。
- **注意力可视化**：可观察模型关注图像的哪些区域，解释性较好。

### 3.8 优势与应用
- 优势：长距离建模能力强；扩展性更好（参数量可轻易扩大）；适合多模态统一建模。
- 应用：图像分类、目标检测（ViT-FRCNN、DETR）、分割（SETR）、视频理解（TimeSformer）。

---

## 4 CLIP

### 4.1 CLIP总览
- **全称**：Contrastive Language–Image Pre-training
- **发布者**：OpenAI（2021）
- **核心思想**：通过对比学习使图像和文本在共享嵌入空间中对齐，实现**零样本迁移**。

### 4.2 模型架构
- **双塔结构**：
  - **图像编码器**：ViT 或 ResNet，将图像映射到d维向量。
  - **文本编码器**：Transformer（类似GPT），将文本映射到同一d维向量。
- **投影头**：两个编码器后各接一个可学习的线性投影层，将特征映射到多模态嵌入空间。
  
### 4.3 代码示例：CLIP 对比损失实现
```python
def contrastive_loss(image_features, text_features, temperature=0.07):
    # image_features, text_features: (batch_size, feature_dim) 已L2归一化
    logits = image_features @ text_features.T / temperature  # (B, B)
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)      # image -> text
    loss_t = F.cross_entropy(logits.T, labels)    # text -> image
    return (loss_i + loss_t) / 2

# 使用示例
image_emb = img_encoder(images)    # (B, D)
text_emb = text_encoder(texts)     # (B, D)
image_emb = F.normalize(image_emb, dim=-1)
text_emb = F.normalize(text_emb, dim=-1)
loss = contrastive_loss(image_emb, text_emb, temperature=0.07)
```

### 4.4 损失函数推导：对称对比损失（InfoNCE）
对于包含 $$N$$ 个图文对的 batch，定义相似度矩阵 $$S_{i,j} = \frac{f(I_i) \cdot g(T_j)}{\tau}$$，其中 \$$tau$$ 是温度参数。

- 图像到文本的损失：
$$
\mathcal{L}_{i2t} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{i,i})}{\sum_{j=1}^{N}\exp(S_{i,j})}
$$
- 文本到图像的损失：
$$
\mathcal{L}_{t2i} = -\frac{1}{N}\sum_{j=1}^{N} \log \frac{\exp(S_{j,j})}{\sum_{i=1}^{N}\exp(S_{i,j})}
$$
总损失：
$$
\mathcal{L} = \frac{1}{2}$$mathcal{L}_{i2t} + \mathcal{L}_{t2i})
$$
等价于最大化正样本对的相似度，同时最小化负样本对的相似度。

### 4.5 训练超参数表（CLIP 原始 ViT‑B/32）

| 超参数             | 典型值                          |
| ------------------ | ------------------------------- |
| 图像编码器         | ViT‑B/32 或 ResNet50x4          |
| 文本编码器         | Transformer (12层, 512宽, 8头)  |
| 嵌入维度           | 512                             |
| 批量大小           | 32768                           |
| 学习率             | 5e-4 (余弦退火)                 |
| 优化器             | AdamW (β1=0.9, β2=0.98, eps=1e-6) |
| 温度参数 τ         | 初始 0.07, 可学习                |
| 训练步数           | 32 epochs on 4B图文对           |
| 混合精度           | FP16                            |

### 4.6 训练方法
- **数据**：4亿个（图像，文本）对，从互联网收集，未经过精细清洗。
- **对比目标**：
  - 给定一个batch内N个图文对，构造N个正样本对（匹配的图文），N² - N个负样本对（不匹配）。
  - 计算相似度矩阵（如余弦相似度），对图像到文本方向和文本到图像方向都做softmax交叉熵损失。
  - 对称对比损失（InfoNCE形式）：
    $$
    \mathcal{L} = \frac{1}{2}$$mathcal{L}_{\text{img2txt}} + \mathcal{L}_{\text{txt2img}})
    $$
- **批量大小**：使用极大batch（32768）以提供丰富负样本。

### 4.7 零样本推理流程
1. 为每个类别构造提示模板（如“A photo of a {class}”）。
2. 将模板句子通过文本编码器得到类别文本向量。
3. 将待分类图像通过图像编码器得到图像向量。
4. 计算图像与所有类别文本向量的相似度，取最高者作为预测。

### 4.8 优势与局限
- **优势**：
  - 零样本分类能力强，可泛化到未见类别。
  - 学习到的特征鲁棒且可迁移。
  - 可作为多模态基础模型，支持图文检索、生成引导等。
- **局限**：
  - 细粒度理解弱（区分“狼”与“狗”）。
  - 对抽象概念（“错误”、“奇怪”）不敏感。
  - 计数、位置关系等能力差。
  - 训练需要海量图文对。

### 4.9 后续改进（如OpenCLIP、SigLIP、BLIP）
- **SigLIP**：使用sigmoid损失替代softmax，允许更大batch独立负采样。
- **BLIP**：在CLIP基础上加入图像描述生成模块和过滤噪声数据机制。
- **ALIGN**：利用更大规模噪声数据（10亿对）训练，刷出新SOTA。

---

## 5 知识蒸馏与DINO

### 5.1 知识蒸馏（Knowledge Distillation, KD）

#### 5.1.1 基本概念
- **定义**：将大型教师模型的知识迁移到小型学生模型，实现压缩加速。
- **软标签**：教师模型输出的类别概率分布（含暗知识，如“猫”与“豹”的相似度）。
- **经典公式**：
  $$
  \mathcal{L} = \alpha \cdot \mathcal{L}_{\text{hard}}(y, p_s) + (1-\alpha) \cdot \mathcal{L}_{\text{soft}}(q_t, q_s)
  $$
  其中 $$q_t, q_s$$ 是经过温度$$T$$平滑后的教师/学生输出分布。

#### 5.1.2 代码示例：知识蒸馏损失
```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    # 软标签损失（KL散度）
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    # 硬标签损失
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss
```

#### 5.1.3 损失函数推导：知识蒸馏
教师输出概率 $$q_i = \frac{\exp(z_i^t/T)}{\sum_j \exp(z_j^t/T)}$$，学生输出 $$p_i$$ 类似。蒸馏损失使用 KL 散度：
$$
\mathcal{L}_{\text{KD}} = T^2 \cdot \text{KL}(q \parallel p) = T^2 \sum_i q_i \log\frac{q_i}{p_i}
$$
乘以 $$T^2$$ 是为了使梯度幅度与温度参数无关。

### 5.2 训练超参数表（DINO ViT‑B/16）

| 超参数      | 典型值                                |
| -------- | ---------------------------------- |
| 教师温度 τ_t | 0.04                               |
| 学生温度 τ_s | 0.1                                |
| 中心更新动量   | 0.9                                |
| 教师EMA动量  | 从0.996线性增至1.0                      |
| 优化器      | AdamW (lr=5e-4, weight decay=0.04) |
| 学习率调度    | 余弦退火，无warmup                       |
| 批量大小     | 1024 (分布式)                         |
| 训练轮数     | 300 epochs on ImageNet             |
| 全局视图尺寸   | 224×224 (2个)                       |
| 局部视图尺寸   | 96×96 (10个)                        |
| 随机深度率    | 0.1 (教师网络无)                        |


#### 5.2.1 视觉中的蒸馏应用
- **DeiT**：使用CNN教师蒸馏ViT学生，使ViT在ImageNet上达到SOTA。
- **分类蒸馏**：比单纯用真值标签训练效果更好。

### 5.3 DINO（DIstillation with NO labels）

#### 5.3.1 总体定位
- **DINO**：一种自监督视觉Transformer训练方法，无标签仅需图像，通过知识蒸馏实现。
- **特点**：无需负样本，无需聚类，直接学习有意义的视觉特征。
#### 5.3.2 代码示例：DINO 核心模块（EMA更新 + 多裁剪损失）
```python
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
        
    def forward(self, student_output, teacher_output):
        # student_output: (B, D), teacher_output: (B, D)
        student_out = student_output / self.student_temp
        student_out = F.log_softmax(student_out, dim=-1)
        
        teacher_out = teacher_output / self.teacher_temp
        teacher_out = F.softmax(teacher_out, dim=-1)
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()
        
        # 更新center (EMA)
        self.center = self.center * self.center_momentum + teacher_output.mean(dim=0, keepdim=True) * (1 - self.center_momentum)
        return loss

# EMA 更新函数
def update_ema(teacher, student, momentum=0.996):
    for param_t, param_s in zip(teacher.parameters(), student.parameters()):
        param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data
```

#### 5.3.3 损失函数推导：DINO 自蒸馏损失
设有全局视图 $$x_g$$ 和局部视图 $$x_l$$，学生网络 $$f_s$$ 和教师网络 $$f_t$$（教师EMA更新）。教师输出概率分布 $$P_t(x_g) = \text{softmax}(f_t(x_g)/\tau_t)$$，学生输出 $$P_s(x_l) = \text{softmax}(f_s(x_l)/\tau_s)$$。训练目标为最小化交叉熵：
$$
\mathcal{L} = - \sum_{k} P_t(x_g)_k \log P_s(x_l)_k
$$
为避免崩溃，教师输出进行中心化$$(P_t = \text{softmax}((f_t - c)/\tau_t)$$）并维持较高的温度 \$$tau_t$$，学生温度 \$$tau_s$$ 较低。

#### 5.3.4 核心思想——自蒸馏
- **学生-教师对称架构**：两个相同网络（学生和教师），教师参数由学生通过**指数移动平均（EMA）** 更新。
- **多视角裁剪**：
  - 全局视图（大裁剪，覆盖率>50%）→ 教师输入
  - 局部视图（小裁剪，覆盖率<50%）→ 学生输入
  - 目标：学生从局部视图预测教师的全局视图表示。

#### 5.3.5 训练目标
- 教师输出概率分布 $$P_t$$（经过centering和sharpening），学生输出 $$P_s$$。
- 最小化交叉熵：\$$mathcal{L} = -P_t \log P_s$$。
- **避免崩溃**：
  - 教师使用EMA + 中心化（centering）抑制特征坍缩。
  - 学生输出经过softmax，无明确正则项。

#### 5.3.6 关键组件
- **EMA更新**：教师参数 \$$theta_t \leftarrow m \theta_t + (1-m) \theta_s$$，m接近1（如0.996~0.999）。
- **中心化**：教师输出减去一个可学习的中心向量，防止某个维度占主导。
- **锐化**：教师输出除以前一温度系数$$T$$（<1），使分布更尖。

#### 5.3.7 重要结论（DINO论文发现）
- **ViT自监督显著优于有监督**：在无标签数据上训练ViT，特征质量甚至超过有监督训练。
- **注意力图自动出现语义分割特性**：ViT的[CLS] token注意力图能自动分离物体前景与背景，无需任何标签。
- **DINO特征可作为通用视觉表示**：适用于分类、检测、分割、深度估计等多种任务。

#### 5.3.8 后续工作
- **DINOv2**：扩大模型与数据（~1.4亿张图），引入判别式自监督损失，得到接近甚至超越CLIP的视觉特征，且无需文本监督。

### 5.4 蒸馏 vs 自监督 vs 对比学习对比

| 方法         | 是否需要标签 | 核心机制             | 代表工作         |
| ------------ | ------------ | -------------------- | ---------------- |
| 知识蒸馏     | 是（教师标签）| 模仿教师输出         | DeiT, DistilBERT |
| 自监督对比   | 否           | 正样本拉近负样本推远 | SimCLR, MoCo     |
| 自监督蒸馏   | 否           | 学生匹配教师视图     | DINO, BYOL       |

---

## 6 总结

### 6.1 知识点回顾
| 章节               | 核心内容                                                                 |
| ------------------ | ------------------------------------------------------------------------ |
| 大模型技术概述     | 定义、Scaling Law、预训练-微调范式、分布式训练、RLHF                     |
| 多模态大模型技术   | 跨模态对齐、融合架构（Flamingo, LLaVA）、多模态指令微调                  |
| 视觉Transformer    | Patch嵌入、全局注意力、DeiT/Swin、数据需求与归纳偏置权衡                 |
| CLIP               | 双塔对比学习、零样本分类、图文对齐、优缺点与改进                         |
| 知识蒸馏与DINO     | 知识迁移（软标签）、自蒸馏（EMA+多裁剪）、无需标签的高质量视觉表示的形成 |

### 6.2 各节损失函数与超参数速查表

| 模型/方法 | 损失函数核心 | 关键超参数 |
|-----------|--------------|-------------|
| GPT       | 交叉熵（自回归） | lr=6e-5, batch=3.2M tokens |
| ViT       | 交叉熵（分类） | lr=3e-3, weight decay=0.3 |
| CLIP      | 对称对比损失 | τ=0.07, batch=32768 |
| LLaVA     | 语言建模损失（仅回答部分） | lr=2e-5, 视觉编码器冻结 |
| 知识蒸馏  | KL散度 + CE | T=3.0, α=0.5 |
| DINO      | 交叉熵（自蒸馏） | τ_t=0.04, τ_s=0.1, EMA=0.996 |

### 6.3 技术脉络与演进逻辑
- 从**单任务CNN**走向**统一ViT**：Transformer打破模态壁垒。
- 从**有监督**走向**自监督+对比**：缓解对标签的依赖，扩大数据规模。
- 从**单模态**走向**多模态**：对齐图文、视频、语音，迈向通用智能。
- 从**任务专用**走向**基础模型+微调/提示**：大幅降低下游任务应用成本。

### 6.4 当前挑战
- **幻觉问题**：大模型生成看似合理但错误的内容，尤其在多模态场景。
- **计算效率**：训练与推理成本极高，需要模型压缩与稀疏化。
- **复杂推理**：多步骤、多模态、需要外部知识的推理仍不足。
- **评估体系**：如何科学评估大模型的理解、生成、推理、安全等能力。

### 6.5 未来趋势
- **更大更强的基础模型**：万亿参数、更长上下文（1M+）。
- **世界模型与具身智能**：多模态大模型直接控制机器人，与物理世界交互。
- **模型混合与路由**：Mixture of Experts (MoE) 使模型更高效。
- **轻量化与端侧部署**：通过蒸馏、量化、剪枝让大模型运行在手机/PC。
- **自进化与持续学习**：模型从环境中不断学习新知识而不遗忘。

### 6.6 学习建议
- 动手实现：跑通ViT、CLIP、DINO的官方代码，修改并观察效果。
- 精读论文：推荐阅读《Attention is All You Need》、ViT、CLIP、DINO原文。
- 关注开源项目：OpenCLIP、HuggingFace Transformers、DINOv2。
- 参与讨论与竞赛：如多模态VQA、零样本分类、自监督学习benchmark。
