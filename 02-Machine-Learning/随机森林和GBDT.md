### 什么是随机森林

- 随机森林就是通过集成学习的思想把多棵树集成的一种算法
- 他的基本单元是决策树
- 从直观的角度来解释/每棵决策树都是一个分类器
- 对弈一个输入的样本,N课树会有N个分类结果
- 随机森林集成所有的分类投票结果,把投票次数最多的类别指定为最终的输出



### 随机森林的特点

- 等有效运行在大数据集上
- 能够处理具有高为特征的输入样本/不用降维
- 能够评估各个特征在分类问题上的重要性
- 对于缺省值问题也能获得很好的结果 



### 随机森林的相关基础知识

- 信息熵和信息增益的概念
- 决策树C4.5和ID3
- 集成学习



### 随机森林的生成

- 有了树我们才可以分类/那么我们怎么生成树
- 每棵树的规则
  - 如果训练集为N,对于每棵树,**随机**并且有放回从训练集中取n个训练样本/作为训练集
  - 每棵树训练集都是不同的/而且有重复的训练样本
  - 如果不是有放回的抽样,每棵树的训练样本不一样,那么可能会某棵树的偏差大

- 每个样本的特征维度M,指定一个常数m <<M,**随机**从M个特征中选择m个特征子集,树每一次分裂的时候,从m个特征选择最优的

- 这样的两个随机/让随机森林的分类不容易过拟合,而且对缺省值没那么敏感



### 分类效果

- 森林中任意两棵树:相关性越大,错误率越高
- 每棵树的分类能力:每棵树的分类能力越强,整个森林的错误率越低
- 减少特征m,树的相关性和分类能力也会降低
- 增大特征m,两者都增大,所以m很重要
- 袋外错误率(oob)
  - 要解决怎么选择m,主要就是一句计算袋外错误率
  - 对于每一棵树k,大概有1/3的训练实例没参与第k棵树的生成/这些就是第k棵树的oob样本
- oob估计
  - 对每一个样本,计算它的oob样本作为树对它的分类情况
  - 然后简单多数投票作为这个样本的分类结果
  - 最后用误分个数占样本总数的比率作为随机森林的oob误分率



### 什么是GBDT

- GBDT全称是Gradient Boosting Decision Tree梯度提升树
- GBDT使用的是CART回归树,因为每次迭代要拟合的是梯度值,是连续值所以要回归树
- 对于回归树算法来说最重要的是寻找最佳划分点/那么回归树中的可划分点包含了所有特征的所有可取的值
- 在分类树中最佳划分点的判别标准是熵或者基尼系数/都是用纯度衡量的
- 但是回归树的样本标签是连续数值/所以用熵不适合/取而代之的是平方误差/它能很好的评判拟合程度



### 什么是CART算法

- Classification And Regression Tree/分类回归树
- 是一种二分递归分割技术/把当前样本分割成两个子样本/生成的每个非叶子节点都有两个分支/生成的决策树是简洁的二叉树
- 步骤
  - 把样本递归划分进行建树
  - 用验证数据进行剪枝



### CART原理

- 选择一个自变量/再选择一个值/把空间划分成两个部分/一部分满足/另一部分不满足

- 递归处理/知道吧空间划分完成

  - 根据什么来划分呢?对于一个变量属性/划分点是一对连续变量属性值的终点

  - 假设样本的集合一个属性有个连续的值/那么分裂顶点/每个分裂点为相邻两个连续值的均值

  - 每个属性的划分按照能减少的杂质量来进行排序/杂质的减少量定义为划分前的杂质-划分后的每个结点的杂质量划分所占比率之和

  - $$
    Gini(A)=1-\sum_{i=1}^{C}p_{i}^{2}
    $$

  - 其中$p_{i}$表示属于第i类别的概率,当Gini(A) = 0,所有样本属于同一类,所有类在节点中等概率出现的时候Gini(A)最大化

  - 实际的递归划分过程是这样的：如果当前节点的所有样本都不属于同一类或者只剩下一个样本，那么此节点为非叶子节点，所以会尝试样本的每个属性以及每个属性对应的分裂点，尝试找到杂质变量最大的一个划分，该属性划分的子树即为最优分支。



### 举例说明

![](https://imgkr.cn-bj.ufileos.com/ab6a73cb-4bc0-4dd1-a542-3a7a79008390.png)

- 三个属性/有房情况/婚姻/收入
- 有房情况和婚姻状况是离散的/年收入是连续的
- ![](https://imgkr.cn-bj.ufileos.com/af24f78e-3074-4d95-a214-7cc2c43fe950.png)
- ![](https://imgkr.cn-bj.ufileos.com/58c2f568-a367-42c3-8032-fecd426f4acb.png)

- 最后一个取值现需的属性年收入/采用分裂点进行分裂
- ![](https://imgkr.cn-bj.ufileos.com/67b8c1e1-d046-4da9-b9ad-b875daf9ab52.png)

## GDBT

- GBDT的原理很简单，就是所有弱分类器的结果相加等于预测值，然后下一个弱分类器去拟合误差函数对预测值的残差(这个残差就是预测值与真实值之间的误差)。当然了，它里面的弱分类器的表现形式就是各棵树。

  举一个非常简单的例子，比如我今年30岁了，但计算机或者模型GBDT并不知道我今年多少岁，那GBDT咋办呢？

  - 它会在第一个弱分类器（或第一棵树中）随便用一个年龄比如20岁来拟合，然后发现误差有10岁；
  - 接下来在第二棵树中，用6岁去拟合剩下的损失，发现差距还有4岁；
  - 接着在第三棵树中用3岁拟合剩下的差距，发现差距只有1岁了；
  - 最后在第四课树中用1岁拟合剩下的残差，完美。
  - 最终，四棵树的结论加起来，就是真实年龄30岁（实际工程中，gbdt是计算负梯度，用负梯度近似残差）。

  **为何gbdt可以用用负梯度近似残差呢？**

  回归任务下，GBDT 在每一轮的迭代时对每个样本都会有一个预测值，此时的损失函数为均方差损失函数，

  ![](https://imgkr.cn-bj.ufileos.com/1fa57a12-c29f-4466-bb66-fdd7c5332ddc.png)

  那此时的负梯度是这样计算的

  ![](https://imgkr.cn-bj.ufileos.com/c61cf461-4b07-40f9-8051-618e77aab68d.png)

  所以，当损失函数选用均方损失函数是时，每一次拟合的值就是（真实值 - 当前模型预测的值），即残差。此时的变量是$y^{i}$，即“当前预测模型的值”，也就是对它求负梯度。

  **训练过程**

  简单起见，假定训练集只有4个人：A,B,C,D，他们的年龄分别是14,16,24,26。其中A、B分别是高一和高三学生；C,D分别是应届毕业生和工作两年的员工。如果是用一棵传统的回归决策树来训练，会得到如下图所示结果：

  ![](https://imgkr.cn-bj.ufileos.com/6d969a59-ff5b-4a13-8017-c961af3cb90f.png)

  现在我们使用GBDT来做这件事，由于数据太少，我们限定叶子节点做多有两个，即每棵树都只有一个分枝，并且限定只学两棵树。我们会得到如下图所示结果：

  ![](https://imgkr.cn-bj.ufileos.com/50a17f57-c7db-4310-b1c0-78a34237ac8b.png)

  在第一棵树分枝和图1一样，由于A,B年龄较为相近，C,D年龄较为相近，他们被分为左右两拨，每拨用平均年龄作为预测值。

  - 此时计算残差（残差的意思就是：A的实际值 - A的预测值 = A的残差），所以A的残差就是实际值14 - 预测值15 = 残差值-1。
  - 注意，A的预测值是指前面所有树累加的和，这里前面只有一棵树所以直接是15，如果还有树则需要都累加起来作为A的预测值。

  然后拿它们的残差-1、1、-1、1代替A B C D的原值，到第二棵树去学习，第二棵树只有两个值1和-1，直接分成两个节点，即A和C分在左边，B和D分在右边，经过计算（比如A，实际值-1 - 预测值-1 = 残差0，比如C，实际值-1 - 预测值-1 = 0），此时所有人的残差都是0。残差值都为0，相当于第二棵树的预测值和它们的实际值相等，则只需把第二棵树的结论累加到第一棵树上就能得到真实年龄了，即每个人都得到了真实的预测值。

  换句话说，现在A,B,C,D的预测值都和真实年龄一致了。Perfect！

  - A: 14岁高一学生，购物较少，经常问学长问题，预测年龄A = 15 – 1 = 14
  - B: 16岁高三学生，购物较少，经常被学弟问问题，预测年龄B = 15 + 1 = 16
  - C: 24岁应届毕业生，购物较多，经常问师兄问题，预测年龄C = 25 – 1 = 24
  - D: 26岁工作两年员工，购物较多，经常被师弟问问题，预测年龄D = 25 + 1 = 26

  所以，GBDT需要将多棵树的得分累加得到最终的预测得分，且每一次迭代，都在现有树的基础上，增加一棵树去拟合前面树的预测结果与真实值之间的残差。

  ## 2. 梯度提升和梯度下降的区别和联系是什么？

  下表是梯度提升算法和梯度下降算法的对比情况。可以发现，两者都是在每 一轮迭代中，利用损失函数相对于模型的负梯度方向的信息来对当前模型进行更 新，只不过在梯度下降中，模型是以参数化形式表示，从而模型的更新等价于参 数的更新。而在梯度提升中，模型并不需要进行参数化表示，而是直接定义在函 数空间中，从而大大扩展了可以使用的模型种类。

  ![](https://imgkr.cn-bj.ufileos.com/e71e7759-aa50-4b73-9713-987190d5b6e7.png)

  ## 3. **GBDT**的优点和局限性有哪些？

  ### 3.1 优点

  1. 预测阶段的计算速度快，树与树之间可并行化计算。
  2. 在分布稠密的数据集上，泛化能力和表达能力都很好，这使得GBDT在Kaggle的众多竞赛中，经常名列榜首。
  3. 采用决策树作为弱分类器使得GBDT模型具有较好的解释性和鲁棒性，能够自动发现特征间的高阶关系。

  ### 3.2 局限性

  1. GBDT在高维稀疏的数据集上，表现不如支持向量机或者神经网络。
  2. GBDT在处理文本分类特征问题上，相对其他模型的优势不如它在处理数值特征时明显。
  3. 训练过程需要串行训练，只能在决策树内部采用一些局部并行的手段提高训练速度。

  ## 4. RF(随机森林)与GBDT之间的区别与联系

  **相同点**：

  - 都是由多棵树组成，最终的结果都是由多棵树一起决定。
  - RF和GBDT在使用CART树时，可以是分类树或者回归树。

  **不同点**：

  - 组成随机森林的树可以并行生成，而GBDT是串行生成
  - 随机森林的结果是多数表决表决的，而GBDT则是多棵树累加之和
  - 随机森林对异常值不敏感，而GBDT对异常值比较敏感
  - 随机森林是减少模型的方差，而GBDT是减少模型的偏差
  - 随机森林不需要进行特征归一化。而GBDT则需要进行特征归一化