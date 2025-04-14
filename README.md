# 🌟 **Daily-DeepLearning** 🌟  

### 

------------

欢迎来到 **Daily-DearnLearning**，原本这是一个为了自己打造的**深度学习知识库**（⬇️滑到最下面或者看目录，可以看以前的和机器学习、深度学习相关的内容），涵盖[计算机基础课程](07-BaseClass/)、[Python快速入门](01-Python/)、[数据科学包的使用](05-Machine-Learning-Code/数据分析工具/)、[机器学习](02-Machine-Learning/)、[深度学习](03-Deep-Learning/)、[自然语言处理](04-NLP/)、[LLM](08-LLM/)等。

---

### 2017年-Attention is All you need

**出现的背景**

要说LLM，大家第一反应应该都是《Attention is all you need》这篇论文。在那之前，因为李飞飞教授推动的ImageNet数据集、GPU算力的提升，那时像CNN刚刚开始流行起来，是用Tensoflow或者Theano写一个手写数字识别。后来开始有人在NLP领域，用word2vec和LSTM的组合，在很多领域里做到SOTA的效果。后来就是2017年，由Google团队提出的这篇里程碑式的论文。



**创新点**

1. 模型的主体结构不再是CNN、RNN的变种，用了用**self-Attention**为主的Transformer结构，所以这篇论文的标题才会说这是all you need嘛。这种方法解决了无法并行计算并且长距离捕捉予以的问题。[自注意力机制解析](08-LLM/Attentionisallyouneed/selfattention.md)

2. 多头注意力机制**Multi-Head Attention**，把输入映射到多个不同的空间，并行计算注意力，有点像CV的RGB、进阶版的词向量的感觉，捕捉不同维度的语义信息，比如语法、语意、上下文信息等等。[多头注意力机制解析](08-LLM/Attentionisallyouneed/multihead.md)

3. 用了位置编码Positional Encoding，这个点很巧妙，因为以前RNN、Lstm输入的时候是顺序输入的，虽然慢，但是正是这种序列化的表示。[位置编码机制解析](08-LLM/Attentionisallyouneed/positionalencoding.md)

PS：如果对编码不太了解，可以看看以前的编码方式，比如机器学习时期的[词袋模型TF-IDF](04-NLP/词袋模型-TFIDFmd) 或者深度学习时期的[词向量](03-Deep-Learning/Word2Vec.md)

---

[核心解析](08-LLM/Attentionisallyouneed/核心解析.md) | [论文链接](08-LLM/Attentionisallyouneed/attentionisallyouneed.pdf)  | [简单例子](08-LLM/Attentionisallyouneed/example.md) | [自注意力机制](08-LLM/Attentionisallyouneed/selfattention.md) | [多头注意力](08-LLM/Attentionisallyouneed/multihead.md) | [位置编码](08-LLM/Attentionisallyouneed/positionalencoding.md) | [Harvard NLP PyTorch实现Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) | [Transformer复现](08-LLM/Attentionisallyouneed/Transformer_code.md)

------

### 2018年 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**出现的背景**

Bert比较特殊的地方在于采用了**双向上下文建模**，通过掩码语言模型（Masked language Model），同时利用左右两侧上下文，解决传统模型中的单向性问题。还有很重要的一点，从Bert看来是，形成了“预训练+微调”的新范式，统一了多种NLP任务的框架，仅需在预训练模型基础上添加简单任务头即可适配下游任务。当时在11项NLP任务上刷新SOTA，开启了大规模预训练模型（Pre-trained Language Model, PLM）时代。[Bert解析](08-LLM/Bert/核心解析.md)

---

**直接看论文**

论文戳这里：[*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805)

---

**到底创新了什么**

1. 输入内容的表示方式：词嵌入（WordPiece） + 位置嵌入（Position） + 段落嵌入（Segment）

---

**举个例子**

Bert的双向上下文建模改变了文本表示的学习方式，通过Transformer的编码器结构同时捕捉文本中每个词的左右两侧上下文信息，从而更全面地理解语言语义。

输入表示上使用词嵌入（WordPiece） + 位置嵌入（Position） + 段落嵌入（Segment）

```markdown
整体模型:
输入层 → Embedding → Transformer Encoder × L → 输出层
```

- 输入层：将文本转化成768维向量（BERT-base）
- Encoder层数：BERT-base（L=12）、BERT-large（L=24）
- 输出层：根据任务选择输出形式（如 `[CLS]` 向量用于分类）

```markdown
单层 Encoder 的详细计算流程：
输入向量 → LayerNorm → 多头自注意力 → 残差连接 → LayerNorm → 前馈网络 → 残差连接 → 输出
```

我们可以使用BLEU数据集进行简单的复现[Bert复现](08-LLM/Bert/Bert_code.md)

---

### 2018年 GPT1：Generative Pre-trained Transformer

---

### 2018年  ELMo：Embeddings from Language Models

**出现的背景**

ELMo这个工作主要还是对词向量的改进，从静态的词向量转变成动态词向量，从而提升各项NLP任务上的性能。虽然和GPT、BERT在同一年的工作，但其实应该放在这两项工作前面的，从马后炮的角度来说，主要用的还是双向LSTM，相较于Transformer这样支持并行计算的架构，再配合上MLM来捕捉双向上下文，现在看起来更像是上一代的产物了。但对比起word2vec、GloVe等静态词向量，还是不知道高到哪里去了。[ELMo解析](08-LLM/ELMo/核心解析.md)

---

**直接看论文**

论文戳这里[ELMo:Embeddings from Language Models](https://arxiv.org/abs/1802.05365)

---

**到底创新了什么**

More....

---

## 🌐 **目录**  

### 🖥️ **计算机基础课程**  
**数据结构**  

- [基本概念和算法评价](07-BaseClass/Ds/01基本概念和算法评价.md)  
- [线性表](07-BaseClass/Ds/02线性表.md)  
- [栈和队列](07-BaseClass/Ds/03栈和队列.md)  
- [树和二叉树](07-BaseClass/Ds/04树和二叉树.md)  
- [图](07-BaseClass/Ds/05图.md)  
- [查找](07-BaseClass/Ds/06查找.md)  
- [排序](07-BaseClass/Ds/07排序.md)  

**操作系统**  
- [操作系统的基本概念](07-BaseClass/Os/01操作系统的基本概念.md)  
- [操作系统的发展和分类](07-BaseClass/Os/02操作系统的发展和分类.md)  
- [操作系统的运行环境](07-BaseClass/Os/03操作系统的运行环境.md)  
- [进程和线程](07-BaseClass/Os/04进程与线程.md)  
- [处理机调度](07-BaseClass/Os/05处理机调度.md)  
- [进程同步](07-BaseClass/Os/06进程同步.md)  
- [死锁](07-BaseClass/Os/07死锁.md)  
- [内容管理概念](07-BaseClass/Os/08内容管理概念.md)  
- [虚拟内存管理](07-BaseClass/Os/09虚拟内存管理.md)  
- [文件系统基础](07-BaseClass/Os/10文件系统基础.md)  

**计算机网络**  
- [计算机网络概述](07-BaseClass/Cn/01计算机网络概述.md)  
- [计算机网络结构体系](07-BaseClass/Cn/02计算机网络结构体系.md)  
- [通信基础](07-BaseClass/Cn/03通信基础.md)  
- [奈氏准则和香农定理](07-BaseClass/Cn/04奈氏准则和香农定理.md)  
- [传输介质](07-BaseClass/Cn/05传输介质.md)  
- [物理层设备](07-BaseClass/Cn/06物理层设备.md)  
- [数据链路层的功能](07-BaseClass/Cn/07数据链路层的功能.md)  

---

### 🐍 **Python 快速入门**  
**Day01**: [变量、字符串、数字和运算符](01-Python/Day01.md)  
**Day02**: [列表、元组](01-Python/Day02.md)  
**Day03**: [字典、集合](01-Python/Day03.md)  
**Day04**: [条件语句、循环](01-Python/Day04.md)  
**Day05**: [函数的定义与调用](01-Python/Day05.md)  
**Day06**: [迭代、生成器、迭代器](01-Python/Day06.md)  
**Day07**: [高阶函数、装饰器](01-Python/Day07.md)  
**Day08**: [面向对象编程](01-Python/Day08.md)  
**Day09**: [类的高级特性](01-Python/Day09.md)  
**Day10**: [错误处理与调试](01-Python/Day10.md)  
**Day11**: [文件操作](01-Python/Day11.md)  
**Day12**: [多线程与多进程](01-Python/Day12.md)  
**Day13**: [日期时间、集合、结构体](01-Python/Day13.md)  
**Day14**: [协程与异步编程](01-Python/Day14.md)  
**Day15**: [综合实践](01-Python/Day15.md)  

---

### 📊 **数据科学包的使用**  
**NumPy**  
- [创建 ndarray](05-Machine-Learning-Code/数据分析工具/Numpy/创建ndarray.md)  
- [数据类型和运算](05-Machine-Learning-Code/数据分析工具/Numpy/数据类型和运算.md)  
- [索引和切片](05-Machine-Learning-Code/数据分析工具/Numpy/索引和切片.md)  
- [矩阵操作](05-Machine-Learning-Code/数据分析工具/Numpy/矩阵操作.md)  

**Pandas**  
- [加载数据](05-Machine-Learning-Code/数据分析工具/Pandas/1_Loading.ipynb)  
- [行列选择](05-Machine-Learning-Code/数据分析工具/Pandas/2_Select_row_and_columns.ipynb)  
- [索引操作](05-Machine-Learning-Code/数据分析工具/Pandas/3_Set_reset_use_indexes.ipynb)  
- [数据过滤](05-Machine-Learning-Code/数据分析工具/Pandas/4_Filtering.ipynb)  
- [更新行列](05-Machine-Learning-Code/数据分析工具/Pandas/5_update_rows_columns.ipynb)  
- [数据排序](05-Machine-Learning-Code/数据分析工具/Pandas/7_sort_data.ipynb)  
- [数据聚合](05-Machine-Learning-Code/数据分析工具/Pandas/8_Grouping_Aggregating.ipynb)  
- [数据清洗](05-Machine-Learning-Code/数据分析工具/Pandas/9_Cleaning_Data.ipynb)  
- [时间数据处理](05-Machine-Learning-Code/数据分析工具/Pandas/10_WorkingWithDatesAndTimeSertesData.ipynb)  

**Matplotlib**  
- [直线图](05-Machine-Learning-Code/数据分析工具/Matplotlib/1_creating_and_customizing_plots.ipynb)  
- [柱状图](05-Machine-Learning-Code/数据分析工具/Matplotlib/2_Bar_charts.ipynb)  
- [饼状图](05-Machine-Learning-Code/数据分析工具/Matplotlib/3_Pie.ipynb)  
- [堆叠图](05-Machine-Learning-Code/数据分析工具/Matplotlib/4_stack.ipynb)  
- [填充图](05-Machine-Learning-Code/数据分析工具/Matplotlib/5_Line_Filling_Area.ipynb)  
- [直方图](05-Machine-Learning-Code/数据分析工具/Matplotlib/6_histograms.ipynb)  
- [散点图](05-Machine-Learning-Code/数据分析工具/Matplotlib/7_Scatter.ipynb)  
- [时序图](05-Machine-Learning-Code/数据分析工具/Matplotlib/8_Time_Series_Data.ipynb)  
- [子图](05-Machine-Learning-Code/数据分析工具/Matplotlib/10_subplot.ipynb)  

---

### 🤖 **机器学习理论与实战**  
**理论**  
- [逻辑回归](02-Machine-Learning/逻辑回归.md)  
- [EM 算法](02-Machine-Learning/EM算法.md)  
- [集成学习](02-Machine-Learning/集成学习入门.md)  
- [随机森林与 GBDT](02-Machine-Learning/随机森林和GBDT.md)  
- [ID3/C4.5 算法](02-Machine-Learning/ID3和C4.5算法.md)  
- [K-means](02-Machine-Learning/K-means.md)  
- [K 最近邻](02-Machine-Learning/K最近邻.md)  
- [贝叶斯](02-Machine-Learning/贝叶斯.md)  
- [XGBoost/LightGBM](02-Machine-Learning/XgBoost和LightGBM.md)  
- [Gradient Boosting](02-Machine-Learning/Gradient_Boosting.md)  
- [Boosting Tree](https://mp.weixin.qq.com/s/Cdi0CcWDLgS6Kk7Kx71Vaw)  
- [回归树](https://mp.weixin.qq.com/s/XiTH-8FY5Aw-p_1Ifhx4oQ)  
- [XGBoost](02-Machine-Learning/XgBoost.md)  
- [GBDT 分类](02-Machine-Learning/GBDT分类.md)  
- [GBDT 回归](02-Machine-Learning/GBDT回归.md)  
- [LightGBM](02-Machine-Learning/LightGBM.md)  
- [CatBoost](02-Machine-Learning/CatBoost.md)  

**实战**  
- **NumPy 实战**：[创建 ndarray](05-Machine-Learning-Code/数据分析工具/Numpy/创建ndarray.md)  
- **Pandas 实战**：[加载数据](05-Machine-Learning-Code/数据分析工具/Pandas/1_Loading.ipynb)  
- **Matplotlib 实战**：[直线图](05-Machine-Learning-Code/数据分析工具/Matplotlib/1_creating_and_customizing_plots.ipynb)  

---

### 🏊‍♀️ **深度学习理论与实战**  
**理论**  
- [Word2Vec](03-Deep-Learning/Word2Vec.md)  
- [BatchNorm](03-Deep-Learning/BatchNorm.md)  
- [Dropout](03-Deep-Learning/Dropout.md)  
- [CNN](03-Deep-Learning/CNN.md)  
- [RNN](03-Deep-Learning/RNN.md)  
- [LSTM](03-Deep-Learning/LSTM.md)  
- [Attention](03-Deep-Learning/Attention.md)  
- [ELMo](03-Deep-Learning/ELMo.md)  
- [Transformer](03-Deep-Learning/Transformer.md)  
- [BERT](03-Deep-Learning/BERT.md)  
- [ALBERT](03-Deep-Learning/ALBERT.md)  
- [XLNet](03-Deep-Learning/XLNet.md)  

**实战**  
- **TensorFlow**  
  - [Hello World](06-Deep-Learning-Code/Tensorflow/Helloworld.md)  
  - [线性回归](06-Deep-Learning-Code/Tensorflow/linear_regression.md)  
  - [逻辑回归](06-Deep-Learning-Code/Tensorflow/logistic_regression.md)  
  - [基本图像分类](06-Deep-Learning-Code/Tensorflow/基本图像分类.ipynb)  
- **PyTorch**  
  - [入门](06-Deep-Learning-Code/pytorch/gettingstart.md)  
  - [自动求导](06-Deep-Learning-Code/pytorch/autograd.ipynb)  
  - [神经网络](06-Deep-Learning-Code/pytorch/NeuralNetworks.ipynb)  

---

### 🀄 **NLP 相关**  
- [Word2Vec](03-Deep-Learning/Word2Vec.md)  
- [LSTM](03-Deep-Learning/LSTM.md)  
- [ELMo](03-Deep-Learning/ELMo.md)  
- [ALBERT](03-Deep-Learning/ALBERT.md)  
- [XLNet](03-Deep-Learning/XLNet.md)  

---

### 📫 **联系我们**  

如果你有任何问题或建议，欢迎通过以下方式联系我们：  

- **邮箱**：[lauzanhing@gmail.com](mailto:lauzanhing@gmail.com)  
- **GitHub Issues**：[https://github.com/yourusername/Daily-DearnLearning/issues](https://github.com/yourusername/Daily-DearnLearning/issues)  

---
