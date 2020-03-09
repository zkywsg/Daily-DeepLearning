来源:人人都懂EM算法(知乎)

**估计有很多入门机器学习的同学在看到EM算法的时候会有种种疑惑：EM算法到底是个什么玩意？它能做什么？它的应用场景是什么？网上的公式推导怎么看不懂？**

**下面我会从一个案例开始讲解极大似然估计，然后过渡到EM算法，讲解EM算法到底是个什么玩意儿以及它的核心的idea是什么。之后讲解EM算法的推导公式，鉴于网上很多博客文章都是直接翻译吴恩达的课程笔记内容，有很多推导步骤都是跳跃性的，我会把这些中间步骤弥补上，让大家都能看懂EM算法的推导过程。最后以一个二硬币模型作为EM算法的一个实例收尾。希望阅读本篇文章之后能对EM算法有更深的了解和认识。**

**极大似然和EM(Expectation Maximization)算法，与其说是一种算法，不如说是一种解决问题的思想，解决一类问题的框架，和线性回归，逻辑回归，决策树等一些具体的算法不同，极大似然和EM算法更加抽象，是很多具体算法的基础。**

## 1. 从极大似然到EM

## 1.1 极大似然

## 1.1.1 问题描述

假设我们需要调查我们学校学生的身高分布。我们先假设学校所有学生的身高服从正态分布$N(\mu,\sigma^{2})$ 。(**注意：极大似然估计的前提一定是要假设数据总体的分布，如果不知道数据分布，是无法使用极大似然估计的**)，这个分布的均值 $\mu$ 和方差$\sigma^{2}$ 未知，如果我们估计出这两个参数，那我们就得到了最终的结果。那么怎样估计这两个参数呢？

学校的学生这么多，我们不可能挨个统计吧？这时候我们需要用到概率统计的思想，也就是抽样，根据样本估算总体。假设我们随机抽到了 200 个人（也就是 200 个身高的样本数据，为了方便表示，下面“人”的意思就是对应的身高）。然后统计抽样这 200 个人的身高。根据这 200 个人的身高估计均值 $\mu$  和方差 $\sigma^{2}$  。

用数学的语言来说就是：为了统计学校学生的身高分布，我们独立地按照概率密度$p(x|\theta)$ 抽取了 200 个（身高），组成样本集 $X=x_{1},x_{2},...,x_{N}$ (其中$x_{i}$表示抽到的第$i$ 个人的身高，这里 N 就是 200，表示样本个数)，我们想通过样本集 X 来估计出总体的未知参数$\theta$  。这里概率密度$p(x|\theta)$  服从正态分布 $N(\mu,\sigma^{2})$ ，其中的未知参数是 $\theta = [\mu,\sigma]^{T}$。        

那么问题来了怎样估算参数$\theta$ 呢？

## 1.1.2 估算参数

我们先回答几个小问题：

**问题一：抽到这 200 个人的概率是多少呢？**

由于每个样本都是独立地从$p(x|\theta) $ 中抽取的，换句话说这 200 个学生随便捉的，他们之间是没有关系的，即他们之间是相互独立的。假如抽到学生 A（的身高）的概率是$p(x_{A}|\theta)$  *，*抽到学生B的概率是$p(x_{B}|\theta)$  ，那么同时抽到男生 A 和男生 B 的概率是 $p(x_{A}|\theta)$x $p(x_{B}|\theta)$，同理，我同时抽到这 200 个学生的概率就是他们各自概率的乘积了，即为他们的联合概率，用下式表示：

![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%29+%3D+L%28x_1%2C+x_2%2C+%5Ccdots+%2C+x_n%3B+%5Ctheta%29+%3D+%5Cprod+%5Climits+_%7Bi%3D1%7D%5E%7Bn%7Dp%28x_i%7C%5Ctheta%29%2C+%5Cquad+%5Ctheta+%5Cin+%5CTheta+%5C%5C)
n 为抽取的样本的个数，本例中 $n=200$ ，这个概率反映了，在概率密度函数的参数是$\theta$时，得到 X 这组样本的概率。上式中等式右侧只有$\theta$ 是未知数，所以 L 是 $\theta$ 的函数。

这个函数反映的是在不同的参数$\theta$取值下，取得当前这个样本集的可能性，因此称为参数$\theta$ 相对于样本集 X 的似然函数（likelihood function），记为$L(\theta)$ 。

对 L 取对数，将其变成连加的，称为对数似然函数，如下式：
 ![[公式]](https://www.zhihu.com/equation?tex=H%28%5Ctheta%29+%3D+%5Ctext%7Bln%7D+%5C+L%28%5Ctheta%29+%3D+%5Ctext%7Bln%7D+%5Cprod+%5Climits+_%7Bi%3D1%7D%5E%7Bn%7Dp%28x_i%7C%5Ctheta%29+%3D+%5Csum+%5Climits+_%7Bi%3D1%7D%5E%7Bn%7D%5Ctext%7Bln%7D+p%28x_i%7C%5Ctheta%29+%5C%5C)

**Q：这里为什么要取对数？**

- ​    取对数之后累积变为累和，求导更加方便
- ​    概率累积会出现数值非常小的情况，比如1e-30，由于计算机的精度是有限的，无法识别这一类数据，取对数之后，更易于计算机的识别(1e-30以10为底取对数后便得到-30)。

**问题二：学校那么多学生，为什么就恰好抽到了这 200 个人 ( 身高) 呢？**

在学校那么学生中，我一抽就抽到这 200 个学生（身高），而不是其他人，那是不是表示在整个学校中，这 200 个人（的身高）出现的概率极大啊，也就是其对应的似然函数$L(\theta)$ 极大，即

![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%5Ctheta+%3D+%5Ctext%7Bargmax%7D+%5C+L%28%5Ctheta%29+%5C%5C)

$\hat{\theta}$ 这个叫做$\theta$  的极大似然估计量，即为我们所求的值。

**问题三：那么怎么极大似然函数？**

求$L(\theta)$  对所有参数的偏导数，然后让这些偏导数为 0，假设有$n$ 个参数，就有$n$  个方程组成的方程组，那么方程组的解就是似然函数的极值点了，从而得到对应的$\theta$  了。

## 1.1.3 极大似然估计总结

极大似然估计你可以把它看作是一个反推。多数情况下我们是根据已知条件来推算结果，而极大似然估计是已经知道了结果，然后寻求使该结果出现的可能性极大的条件，以此作为估计值。

比如说，

- 假如一个学校的学生男女比例为 9:1 (条件)，那么你可以推出，你在这个学校里更大可能性遇到的是男生 (结果)；
- 假如你不知道那女比例，你走在路上，碰到100个人，发现男生就有90个 (结果)，这时候你可以推断这个学校的男女比例更有可能为 9:1 (条件)，这就是极大似然估计。

极大似然估计，只是一种概率论在统计学的应用，它是参数估计的方法之一。说的是已知某个随机样本满足某种概率分布，但是其中具体的参数不清楚，通过若干次试验，观察其结果，利用结果推出参数的大概值。

极大似然估计是建立在这样的思想上：已知某个参数能使这个样本出现的概率极大，我们当然不会再去选择其他小概率的样本，所以干脆就把这个参数作为估计的真实值。

## 1.1.4 求极大似然函数估计值的一般步骤：

（1）写出似然函数；

（2）对似然函数取对数，并整理；

（3）求导数，令导数为 0，得到似然方程；

（4）解似然方程，得到的参数。

## 1.1.5 极大似然函数的应用

**应用一 ：回归问题中的极小化平方和** （极小化代价函数）

假设线性回归模型具有如下形式: ![[公式]](https://www.zhihu.com/equation?tex=h%28x%29+%3D+%5Csum+%5Climits+_%7Bi%3D1%7D%5E%7Bd%7D+%5Ctheta_jx_j+%2B+%5Cepsilon+%3D+%5Ctheta%5ETx+%2B+%5Cepsilon)，如何求 $\theta$  呢？

- 最小二乘估计：最合理的参数估计量应该使得模型能最好地拟合样本数据，也就是估计值和观测值之差的平方和最小，其推导过程如下所示：

![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29%3D+%5Csum+%5Climits+_%7Bi%3D1%7D%5En+%28h_%7B%5Ctheta%7D%28x_i%29-+y_i%29+%5E2+%5C%5C)

​        求解方法是通过梯度下降算法，训练数据不断迭代得到最终的值。  

- 极大似然法：最合理的参数估计量应该使得从模型中抽取 m 组样本观测值的概率极大，也就是似然函数极大。
   ![[公式]](https://www.zhihu.com/equation?tex=p%28y_i%7Cx_i%3B%5Ctheta%29+%3D+%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D%5Csigma%7Dexp%28-%5Cfrac%7B%28y_i-%5Ctheta%5ETx_%7Bi%7D%29%5E2%7D%7B2%5Csigma%5E2%7D%29+%5C%5C+%5Cbegin+%7Balign%2A%7DL%28%5Ctheta%29+%26%3D+%5Cprod+%5Climits+_%7Bi%3D1%7D%5Emp%28y_i%7Cx_i%3B%5Ctheta%29+%5C%5C+%26%3D+%5Cprod+%5Climits+_%7Bi%3D1%7D%5Em%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D%5Csigma%7Dexp%28-%5Cfrac%7B%28y_i-%5Ctheta%5ETx_%7Bi%7D%29%5E2%7D%7B2%5Csigma%5E2%7D%29%5Cend%7Balign%2A%7D+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin+%7Balign%2A%7DH%28%5Ctheta%29+%26%3D+log%28L%28%5Ctheta%29%29+%5C%5C+%26%3D+%5Ctext%7Blog%7D%5C+%5Cprod+%5Climits+_%7Bi%3D1%7D%5Em%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D%5Csigma%7Dexp%28-%5Cfrac%7B%28y_i-%5Ctheta%5ETx_%7Bi%7D%29%5E2%7D%7B2%5Csigma%5E2%7D%29+%5C%5C%26%3D+%5Csum+%5Climits+_%7Bi%3D1%7D%5Em%28+%5Ctext%7Blog%7D%5C+%5Cfrac%7B1%7D%7B%5Csqrt%7B2+%5Cpi%7D%5Csigma%7Dexp%28-%5Cfrac%7B%28y_i-%5Ctheta%5ETx_%7Bi%7D%29%5E2%7D%7B2%5Csigma%5E2%7D%29%29+%5C%5C+%26+%3D+-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D+%5Csum+%5Climits+_%7Bi%3D1%7D%5Em%28y_i-%5Ctheta%5ETx_%7Bi%7D%29%5E2+-+m%5Ctext%7Bln%7D%5C+%5Csigma+%5Csqrt%7B2%5Cpi%7D+%5Cend%7Balign%2A%7D+%5C%5C)

 令 ![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29+%3D+%5Cfrac%7B1%7D%7B2%7D+%5Csum+%5Climits+_%7Bi%3D1%7D%5Em%28y_i-%5Ctheta%5ETx_%7Bi%7D%29%5E2) 则![[公式]](https://www.zhihu.com/equation?tex=arg+%5Cmax+%5Climits_%7B%5Ctheta%7D+H%28%5Ctheta%29+%5CLeftrightarrow+arg+%5Cmin+%5Climits_%7B%5Ctheta%7D+J%28%5Ctheta%29+) ， 即将极大似然函数等价于极小化代价函数。

这时可以发现，此时的极大化似然函数和最初的最小二乘损失函数的估计结果是等价的。但是要注意这两者只是恰好有着相同的表达结果，原理和出发点完全不同。

**应用二：分类问题中极小化交叉熵** （极小化代价函数）

在分类问题中，交叉熵的本质就是似然函数的极大化，逻辑回归的假设函数为：
 ![[公式]](https://www.zhihu.com/equation?tex=h%28x%29+%3D+%5Chat+y+%3D+%5Cfrac+1+%7B1%2Be%5E%7B-%5Ctheta%5ETx+%2B+b%7D%7D+%5C%5C)
根据之前学过的内容我们知道 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+y+%3D+p%28y%3D1%7Cx%2C+%5Ctheta%29) ，

当  $y=1$时， ![[公式]](https://www.zhihu.com/equation?tex=p_1+%3D+p%28y+%3D+1%7Cx%2C%5Ctheta%29+%3D+%5Chat+y)

当  $y=0$ 时，![[公式]](https://www.zhihu.com/equation?tex=p_0+%3D+p%28y+%3D+0%7Cx%2C%5Ctheta%29+%3D+1-+%5Chat+y)

合并上面两式子，可以得到

![[公式]](https://www.zhihu.com/equation?tex=p%28y%7Cx%EF%BC%8C+%5Ctheta%29+%3D+%5Chat+y%5Ey%281-+%5Chat+y%29%5E%7B1-+y%7D+%5C%5C) ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin+%7Balign%2A%7DL%28%5Ctheta%29+%26%3D+%5Cprod+%5Climits+_%7Bi%3D1%7D%5Emp%28y_i%7Cx_i%3B%5Ctheta%29+%5C%5C+%26%3D+%5Cprod+%5Climits+_%7Bi%3D1%7D%5Em%5Chat+y_i%5E%7By_i%7D%281-+%5Chat+y_i%29%5E%7B1-+y_i%7D%5Cend%7Balign%2A%7D+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin+%7Balign%2A%7DH%28%5Ctheta%29+%26%3D%5Ctext%7Blog%7D%28L%28%5Ctheta%29%29+%5C%5C+%26%3D+%5Ctext%7Blog%7D%5Cprod+%5Climits+_%7Bi%3D1%7D%5Em%5Chat+y_i%5E%7By_i%7D%281-+%5Chat+y_i%29%5E%7B1-+y_i%7D+%5C%5C%26%3D+%5Csum+%5Climits+_%7Bi%3D1%7D%5Em+%5Ctext%7Blog%7D%5C+%5Chat+y_i%5E%7By_i%7D%281-+%5Chat+y_i%29%5E%7B1-+y_i%7D+%5C%5C+%26+%3D%5Csum+%5Climits+_%7Bi%3D1%7D%5Em+y_i%5C+%5Ctext%7Blog%7D%5C+%5Chat+y_i+%2B+%281-y_i%29%5C+%5Ctext%7Blog%7D%5C+%281+-+%5Chat+y_i%29+%5Cend%7Balign%2A%7D+%5C%5C)

令 ![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29+%3D+-H%28%5Ctheta%29+%3D+-%5Csum+%5Climits+_%7Bi%3D1%7D%5Em+y_i%5C+%5Ctext%7Blog%7D%5C+%5Chat+y_i+%2B+%281-y_i%29%5C+%5Ctext%7Blog%7D%5C+%281+-+%5Chat+y_i%29) 则 ![[公式]](https://www.zhihu.com/equation?tex=arg+%5Cmax+%5Climits_%7B%5Ctheta%7D+H%28%5Ctheta%29+%5CLeftrightarrow+arg+%5Cmin+%5Climits_%7B%5Ctheta%7D+J%28%5Ctheta%29+) ， 即将极大似然函数等价于极小化代价函数。

## 1.2 EM算法

## 1.2.1 问题描述

上面我们先假设学校所有学生的身高服从正态分布$N(\mu,\sigma^{2})$ 。实际情况并不是这样的，男生和女生分别服从两种不同的正态分布，即男生 $N(\mu_{1},\sigma^{2}_{1})$ ，女生 $N(\mu_{2},\sigma^{2}_{2})$ ，(**注意：EM算法和极大似然估计的前提是一样的，都要假设数据总体的分布，如果不知道数据分布，是无法使用EM算法的**)。那么该怎样评估学生的身高分布呢？

简单啊，我们可以随便抽 100 个男生和 100 个女生，将男生和女生分开，对他们单独进行极大似然估计。分别求出男生和女生的分布。

假如某些男生和某些女生好上了，纠缠起来了。咱们也不想那么残忍，硬把他们拉扯开。这时候，你从这 200 个人（的身高）里面随便给我指一个人（的身高），我都无法确定这个人（的身高）是男生（的身高）还是女生（的身高）。用数学的语言就是，抽取得到的每个样本都不知道是从哪个分布来的。那怎么办呢？

## 1.2.2 EM 算法

这个时候，对于每一个样本或者你抽取到的人，就有两个问题需要估计了，一是这个人是男的还是女的，二是男生和女生对应的身高的正态分布的参数是多少。这两个问题是相互依赖的：

- 当我们知道了每个人是男生还是女生，我们可以很容易利用极大似然对男女各自的身高的分布进行估计。
- 反过来，当我们知道了男女身高的分布参数我们才能知道每一个人更有可能是男生还是女生。例如我们已知男生的身高分布为 $N(\mu_{1}=172,\sigma^{2}_{1}=5^{2})$  ， 女生的身高分布为 $N(\mu_{2}=162,\sigma^{2}_{1}=5^{2})$  ， 一个学生的身高为180，我们可以推断出这个学生为男生的可能性更大。

但是现在我们既不知道每个学生是男生还是女生，也不知道男生和女生的身高分布。这就成了一个先有鸡还是先有蛋的问题了。鸡说，没有我，谁把你生出来的啊。蛋不服，说，没有我，你从哪蹦出来啊。为了解决这个你依赖我，我依赖你的循环依赖问题，总得有一方要先打破僵局，不管了，我先随便整一个值出来，看你怎么变，然后我再根据你的变化调整我的变化，然后如此迭代着不断互相推导，最终就会收敛到一个解（草原上的狼和羊，相生相克）。这就是EM算法的基本思想了。

EM的意思是“**Expectation Maximization**”，具体方法为：

- 先设定男生和女生的身高分布参数(初始值)，例如男生的身高分布为$N(\mu_{1}=172,\sigma^{2}_{1}=5^{2})$  ， 女生的身高分布为$N(\mu_{2}=162,\sigma^{2}_{1}=5^{2})$ ，当然了，刚开始肯定没那么准；
- 然后计算出每个人更可能属于第一个还是第二个正态分布中的（例如，这个人的身高是180，那很明显，他极大可能属于男生），这个是属于Expectation 一步；
- 我们已经大概地按上面的方法将这 200 个人分为男生和女生两部分，我们就可以根据之前说的极大似然估计分别对男生和女生的身高分布参数进行估计（这不变成了**极大**似然估计了吗？**极大即为Maximization**）这步称为 Maximization；
- 然后，当我们更新这两个分布的时候，每一个学生属于女生还是男生的概率又变了，那么我们就再需要调整E步；
- ……如此往复，直到参数基本不再发生变化或满足结束条件为止。

## 1.2.3 总结

上面的学生属于男生还是女生我们称之为隐含参数，女生和男生的身高分布参数称为模型参数。

EM 算法解决这个的思路是使用启发式的迭代方法，既然我们无法直接求出模型分布参数，那么我们可以先猜想隐含参数（EM 算法的 E 步），接着基于观察数据和猜测的隐含参数一起来极大化对数似然，求解我们的模型参数（EM算法的M步)。由于我们之前的隐含参数是猜测的，所以此时得到的模型参数一般还不是我们想要的结果。我们基于当前得到的模型参数，继续猜测隐含参数（EM算法的 E 步），然后继续极大化对数似然，求解我们的模型参数（EM算法的M步)。以此类推，不断的迭代下去，直到模型分布参数基本无变化，算法收敛，找到合适的模型参数。

一个最直观了解 EM 算法思路的是 K-Means 算法。在 K-Means 聚类时，每个聚类簇的质心是隐含数据。我们会假设 K 个初始化质心，即 EM 算法的 E 步；然后计算得到每个样本最近的质心，并把样本聚类到最近的这个质心，即 EM 算法的 M 步。重复这个 E 步和 M 步，直到质心不再变化为止，这样就完成了 K-Means 聚类。

## 2. EM算法推导

## 2.1 基础知识

## 2.1.1 凸函数

设是定义在实数域上的函数，如果对于任意的实数，都有：
$$
f^{''}>=0
$$
那么是凸函数。若不是单个实数，而是由实数组成的向量，此时，如果函数的 Hesse 矩阵是半正定的，即
$$
H^{''}>=0
$$


是凸函数。特别地，如果 $f^{''}>0$ 或者$H^{''}>0$ ，称为严格凸函数。

## 2.1.2 Jensen不等式

如下图，如果函数 ![[公式]](https://www.zhihu.com/equation?tex=f) 是凸函数， ![[公式]](https://www.zhihu.com/equation?tex=x) 是随机变量，有 0.5 的概率是 a，有 0.5 的概率是 b， ![[公式]](https://www.zhihu.com/equation?tex=x) 的期望值就是 a 和 b 的中值了那么：
 ![[公式]](https://www.zhihu.com/equation?tex=E%5Bf%28x%29%5D+%5Cge+f%28E%28x%29%29+%5C%5C)
其中，![[公式]](https://www.zhihu.com/equation?tex=E%5Bf%28x%29%5D+%3D+0.5f%28a%29+%2B+0.5+f%28b%29%EF%BC%8Cf%28E%28x%29%29+%3D+f%280.5a+%2B+0.5b%29) ，这里 a 和 b 的权值为 0.5,  ![[公式]](https://www.zhihu.com/equation?tex=f%28a%29)  与 a 的权值相等，![[公式]](https://www.zhihu.com/equation?tex=f%28b%29) 与 b 的权值相等。

特别地，如果函数 ![[公式]](https://www.zhihu.com/equation?tex=f)  是严格凸函数，当且仅当： ![[公式]](https://www.zhihu.com/equation?tex=p%28x+%3D+E%28x%29%29+%3D+1)  (即随机变量是常量) 时等号成立。

![img](https://pic1.zhimg.com/v2-22d1d68bb9db46d48c1a4c194477427c_b.jpg)

注：若函数  ![[公式]](https://www.zhihu.com/equation?tex=f)  是凹函数，Jensen不等式符号相反。

## 2.1.3 期望

对于离散型随机变量 X 的概率分布为  ![[公式]](https://www.zhihu.com/equation?tex=p_i+%3D+p%5C%7BX%3Dx_i%5C%7D) ，数学期望 ![[公式]](https://www.zhihu.com/equation?tex=E%28X%29)  为：
 ![[公式]](https://www.zhihu.com/equation?tex=E%28X%29+%3D+%5Csum+%5Climits+_i+x_ip_i+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=p_i) 是权值，满足两个条件 ![[公式]](https://www.zhihu.com/equation?tex=1+%5Cge+p_i+%5Cge+0%EF%BC%8C%5Csum+%5Climits+_i+p_i+%3D+1)  。

若连续型随机变量X的概率密度函数为 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) ，则数学期望 ![[公式]](https://www.zhihu.com/equation?tex=E%28X%29) 为：
 ![[公式]](https://www.zhihu.com/equation?tex=E%28X%29+%3D+%5Cint+_+%7B-%5Cinfty%7D+%5E%7B%2B%5Cinfty%7D+xf%28x%29+dx+%5C%5C)
设 ![[公式]](https://www.zhihu.com/equation?tex=Y+%3D+g%28X%29)， 若 ![[公式]](https://www.zhihu.com/equation?tex=X) 是离散型随机变量，则：
 ![[公式]](https://www.zhihu.com/equation?tex=E%28Y%29+%3D+%5Csum+%5Climits+_i+g%28x_i%29p_i+%5C%5C)
若  ![[公式]](https://www.zhihu.com/equation?tex=X)  是连续型随机变量，则：
 ![[公式]](https://www.zhihu.com/equation?tex=E%28X%29+%3D+%5Cint+_+%7B-%5Cinfty%7D+%5E%7B%2B%5Cinfty%7D+g%28x%29f%28x%29+dx+%5C%5C)

## 2.2 EM算法的推导

对于 ![[公式]](https://www.zhihu.com/equation?tex=m) 个相互独立的样本 ![[公式]](https://www.zhihu.com/equation?tex=x%3D%28x%5E%7B%281%29%7D%2Cx%5E%7B%282%29%7D%2C...x%5E%7B%28m%29%7D%29) ，对应的隐含数据 ![[公式]](https://www.zhihu.com/equation?tex=z%3D%28z%5E%7B%281%29%7D%2Cz%5E%7B%282%29%7D%2C...z%5E%7B%28m%29%7D%29) ，此时 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cz%29) 即为完全数据，样本的模型参数为 ![[公式]](https://www.zhihu.com/equation?tex=%CE%B8) , 则观察数据 ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28i%29%7D) 的概率为  ![[公式]](https://www.zhihu.com/equation?tex=P%28x%5E%7B%28i%29%7D%7C%5Ctheta%29) ，完全数据 ![[公式]](https://www.zhihu.com/equation?tex=%28x%5E%7B%28i%29%7D%2Cz%5E%7B%28i%29%7D%29) 的似然函数为 ![[公式]](https://www.zhihu.com/equation?tex=P%28x%5E%7B%28i%29%7D%2Cz%5E%7B%28i%29%7D%7C%5Ctheta%29) 。

假如没有隐含变量 ![[公式]](https://www.zhihu.com/equation?tex=z)，我们仅需要找到合适的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 极大化对数似然函数即可：
 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%3Darg+%5Cmax+%5Climits_%7B%5Ctheta%7DL%28%5Ctheta%29+%3D+arg+%5Cmax+%5Climits_%7B%5Ctheta%7D%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%29+%5C%5C)

增加隐含变量 ![[公式]](https://www.zhihu.com/equation?tex=z) 之后，我们的目标变成了找到合适的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 和 ![[公式]](https://www.zhihu.com/equation?tex=z) 让对数似然函数极大*：*
 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta%2C+z+%3D+arg+%5Cmax+%5Climits_%7B%5Ctheta%2Cz%7DL%28%5Ctheta%2C+z%29+%3D+arg+%5Cmax+%5Climits_%7B%5Ctheta%2Cz%7D%5Csum%5Climits_%7Bi%3D1%7D%5Em+log%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29+%5C%5C)

不就是多了一个隐变量 ![[公式]](https://www.zhihu.com/equation?tex=z) 吗？那我们自然而然会想到分别对未知的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 和 ![[公式]](https://www.zhihu.com/equation?tex=z) 分别求偏导，这样做可行吗？

理论上是可行的，然而如果对分别对未知的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 和 ![[公式]](https://www.zhihu.com/equation?tex=z) 分别求偏导，由于![[公式]](https://www.zhihu.com/equation?tex=+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%29) 是 ![[公式]](https://www.zhihu.com/equation?tex=P%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29) 边缘概率(建议没基础的同学网上搜一下边缘概率的概念)，转化为 ![[公式]](https://www.zhihu.com/equation?tex=+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%29) 求导后形式会非常复杂（可以想象下 ![[公式]](https://www.zhihu.com/equation?tex=log%28f_1%28x%29%2B+f_2%28x%29%2B%E2%80%A6)复合函数的求导) ，所以很难求解得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 和 ![[公式]](https://www.zhihu.com/equation?tex=z) 。那么我们想一下可不可以将加号从 log 中提取出来呢？我们对这个式子进行缩放如下：  ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%5Csum%5Climits_%7Bi%3D1%7D%5Em+log%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29+%26+%3D+%5Csum%5Climits_%7Bi%3D1%7D%5Em+log%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DQ_i%28z%5E%7B%28i%29%7D%29%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D+%5Ctag%7B1%7D+%5C%5C+%26+%5Cgeq+%5Csum%5Climits_%7Bi%3D1%7D%5Em+%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DQ_i%28z%5E%7B%28i%29%7D%29log%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D+%5Ctag%7B2%7D+%5Cend%7Balign%7D)

上面第(1)式引入了一个未知的新的分布 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29)，满足：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum+%5Climits+_z+Q_i%28z%29%3D1%2C0+%5Cle+Q_i%28z%29%5Cle+1+%5C%5C)

第(2)式用到了 Jensen 不等式 (对数函数是凹函数)：


 ![[公式]](https://www.zhihu.com/equation?tex=log%28E%28y%29%29+%5Cge+E%28log%28y%29%29+%5C%5C)
其中：

![[公式]](https://www.zhihu.com/equation?tex=E%28y%29+%3D+%5Csum%5Climits_i%5Clambda_iy_i%2C+%5Clambda_i+%5Cgeq+0%2C+%5Csum%5Climits_i%5Clambda_i+%3D1+)

![[公式]](https://www.zhihu.com/equation?tex=y_i+%3D+%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_i+%3D+Q_i%28z%5E%7B%28i%29%7D%29)

也就是说 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D) 为第 i 个样本*，* ![[公式]](https://www.zhihu.com/equation?tex=+Q_i%28z%5E%7B%28i%29%7D%29) 为第 i 个样本对应的权重，那么：

![[公式]](https://www.zhihu.com/equation?tex=E%28log%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D%29+%3D+%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DQ_i%28z%5E%7B%28i%29%7D%29+log%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D+%5C%5C)

上式我实际上是我们构建了 ![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%2C+z%29) 的下界，我们发现实际上就是 ![[公式]](https://www.zhihu.com/equation?tex=log%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D) 的加权求和，由于上面讲过权值 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29) 累积和为1，因此上式是 ![[公式]](https://www.zhihu.com/equation?tex=log%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D) 的加权平均，也是我们所说的期望，**这就是Expectation的来历啦**。下一步要做的就是寻找一个合适的 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%29) 最优化这个下界(M步)。

假设 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 已经给定，那么 ![[公式]](https://www.zhihu.com/equation?tex=logL%28%5Ctheta%29) 的值就取决于 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=+p%28x%5E%7B%28i%29%7D%2Cz%5E%7B%28i%29%7D%29) 了。我们可以通过调整这两个概率使下界逼近 ![[公式]](https://www.zhihu.com/equation?tex=logL%28%5Ctheta%29) 的真实值，当不等式变成等式时，说明我们调整后的下界能够等价于![[公式]](https://www.zhihu.com/equation?tex=logL%28%5Ctheta%29) 了。由 Jensen 不等式可知，等式成立的条件是随机变量是常数，则有：  ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D+%3Dc+%5C%5C)
其中 c 为常数，对于任意 ![[公式]](https://www.zhihu.com/equation?tex=i)，我们得到：
 ![[公式]](https://www.zhihu.com/equation?tex=%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%3Dc%7BQ_i%28z%5E%7B%28i%29%7D%29%7D+%5C%5C)
方程两边同时累加和：
 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bz%7D+%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%3D+c%5Csum%5Climits_%7Bz%7D+%7BQ_i%28z%5E%7B%28i%29%7D%29%7D+%5C%5C)
由于 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bz%7DQ_i%28z%5E%7B%28i%29%7D%29+%3D1)。 从上面两式，我们可以得到：
 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bz%7D+%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%3D+c+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29+%3D+%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7Bc%7D+%3D+%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7B%5Csum%5Climits_%7Bz%7DP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%3D+%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BP%28x%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%3D+P%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%29+%5C%5C)

其中：

边缘概率公式： ![[公式]](https://www.zhihu.com/equation?tex=P%28x%5E%7B%28i%29%7D%7C%5Ctheta%29+%3D+%5Csum%5Climits_%7Bz%7DP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29)

条件概率公式： ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BP%28x%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%3D+P%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%29)

从上式可以发现 ![[公式]](https://www.zhihu.com/equation?tex=Q%28z%29)是已知样本和模型参数下的隐变量分布。

如果 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29+%3D+P%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%29%29) , 则第 (2) 式是我们的包含隐藏数据的对数似然的一个下界。如果我们能极大化这个下界，则也在尝试极大化我们的对数似然。即我们需要极大化下式：  ![[公式]](https://www.zhihu.com/equation?tex=arg+%5Cmax+%5Climits_%7B%5Ctheta%7D+%5Csum%5Climits_%7Bi%3D1%7D%5Em+%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DQ_i%28z%5E%7B%28i%29%7D%29log%5Cfrac%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D%7BQ_i%28z%5E%7B%28i%29%7D%29%7D+%5C%5C)

至此，我们推出了在固定参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)后分布 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29) 的选择问题， 从而建立了 ![[公式]](https://www.zhihu.com/equation?tex=logL%28%5Ctheta%29) 的下界，这是 E 步，接下来的M 步骤就是固定 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29) 后，调整 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)，去极大化![[公式]](https://www.zhihu.com/equation?tex=logL%28%5Ctheta%29)的下界。

去掉上式中常数的部分 ![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29) ，则我们需要极大化的对数似然下界为：
 ![[公式]](https://www.zhihu.com/equation?tex=arg+%5Cmax+%5Climits_%7B%5Ctheta%7D+%5Csum%5Climits_%7Bi%3D1%7D%5Em+%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DQ_i%28z%5E%7B%28i%29%7D%29log%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%5C%5C)

## 2.3 EM算法流程

现在我们总结下EM算法的流程。

输入：观察数据![[公式]](https://www.zhihu.com/equation?tex=x%3D%28x%5E%7B%281%29%7D%2Cx%5E%7B%282%29%7D%2C...x%5E%7B%28m%29%7D%29)，联合分布 ![[公式]](https://www.zhihu.com/equation?tex=p%28x%2Cz+%7C%5Ctheta%29) ，条件分布 ![[公式]](https://www.zhihu.com/equation?tex=p%28z%7Cx%2C+%5Ctheta%29)， 极大迭代次数 ![[公式]](https://www.zhihu.com/equation?tex=J) 。

1) 随机初始化模型参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的初值  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta%5E0)

2)  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bfor+j+from+1+to+J%7D)：

-  E步：计算联合分布的条件概率期望：

![[公式]](https://www.zhihu.com/equation?tex=Q_i%28z%5E%7B%28i%29%7D%29+%3A%3D+P%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%29%29+%5C%5C)

- M步：极大化 ![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%29) ,得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) :

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%3A+%3D+arg+%5Cmax+%5Climits_%7B%5Ctheta%7D%5Csum%5Climits_%7Bi%3D1%7D%5Em%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DQ_i%28z%5E%7B%28i%29%7D%29log%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%5C%5C)

- 重复E、M步骤直到 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 收敛

输出：模型参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)

## 2.4 EM算法另一种理解

坐标上升法（Coordinate ascent）(**类似于梯度下降法，梯度下降法的目的是最小化代价函数，坐标上升法的目的是最大化似然函数；梯度下降每一个循环仅仅更新模型参数就可以了，EM算法每一个循环既需要更新隐含参数和也需要更新模型参数，梯度下降和坐标上升的详细分析参见**[攀登传统机器学习的珠峰-SVM (下)](https://zhuanlan.zhihu.com/p/36535299))：

![img](https://pic4.zhimg.com/v2-389aa0ac570f105b0e3b77ed0d3cf10b_b.jpg)

图中的直线式迭代优化的路径，可以看到每一步都会向最优值前进一步，而且前进路线是平行于坐标轴的，因为每一步只优化一个变量。

这犹如在x-y坐标系中找一个曲线的极值，然而曲线函数不能直接求导，因此什么梯度下降方法就不适用了。但固定一个变量后，另外一个可以通过求导得到，因此可以使用坐标上升法，一次固定一个变量，对另外的求极值，最后逐步逼近极值。对应到EM上，**E步：**固定 θ，优化Q；**M步：**固定 Q，优化 θ；交替将极值推向极大。

## 2.5 EM算法的收敛性思考

EM算法的流程并不复杂，但是还有两个问题需要我们思考：

1） EM算法能保证收敛吗？

2） EM算法如果收敛，那么能保证收敛到全局极大值吗？　

首先我们来看第一个问题, EM 算法的收敛性。要证明 EM 算法收敛，则我们需要证明我们的对数似然函数的值在迭代的过程中一直在增大。即：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%5E%7Bj%2B1%7D%29+%5Cgeq+%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%5E%7Bj%7D%29+%5C%5C)

由于：

![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%2C+%5Ctheta%5E%7Bj%7D%29+%3D+%5Csum%5Climits_%7Bi%3D1%7D%5Em%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5E%7Bj%7D%29%29log%7BP%28x%5E%7B%28i%29%7D%EF%BC%8C+z%5E%7B%28i%29%7D%7C%5Ctheta%29%7D+%5C%5C)

令：

![[公式]](https://www.zhihu.com/equation?tex=H%28%5Ctheta%2C+%5Ctheta%5E%7Bj%7D%29+%3D+%5Csum%5Climits_%7Bi%3D1%7D%5Em%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5E%7Bj%7D%29%29log%7BP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%29%7D+%5C%5C)

上两式相减得到：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%29+%3D+L%28%5Ctheta%2C+%5Ctheta%5E%7Bj%7D%29+-+H%28%5Ctheta%2C+%5Ctheta%5E%7Bj%7D%29+%5C%5C)

在上式中分别取 ![[公式]](https://www.zhihu.com/equation?tex=%CE%B8) 为 ![[公式]](https://www.zhihu.com/equation?tex=%CE%B8%5Ej) 和 ![[公式]](https://www.zhihu.com/equation?tex=%CE%B8%5E%7Bj%2B1%7D)，并相减得到：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%5E%7Bj%2B1%7D%29+-+%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%5E%7Bj%7D%29+%3D+%5BL%28%5Ctheta%5E%7Bj%2B1%7D%2C+%5Ctheta%5E%7Bj%7D%29+-+L%28%5Ctheta%5E%7Bj%7D%2C+%5Ctheta%5E%7Bj%7D%29+%5D+-%5BH%28%5Ctheta%5E%7Bj%2B1%7D%2C+%5Ctheta%5E%7Bj%7D%29+-+H%28%5Ctheta%5E%7Bj%7D%2C+%5Ctheta%5E%7Bj%7D%29+%5D+%5C%5C)

要证明EM算法的收敛性，我们只需要证明上式的右边是非负的即可。

由于![[公式]](https://www.zhihu.com/equation?tex=%CE%B8%5E%7Bj%2B1%7D)使得![[公式]](https://www.zhihu.com/equation?tex=L%28%CE%B8%2C%CE%B8%5Ej%29)极大，因此有：

![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%5E%7Bj%2B1%7D%2C+%5Ctheta%5E%7Bj%7D%29+-+L%28%5Ctheta%5E%7Bj%7D%2C+%5Ctheta%5E%7Bj%7D%29+%5Cgeq+0+%5C%5C)

而对于第二部分，我们有：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+H%28%5Ctheta%5E%7Bj%2B1%7D%2C+%5Ctheta%5E%7Bj%7D%29+-+H%28%5Ctheta%5E%7Bj%7D%2C+%5Ctheta%5E%7Bj%7D%29+%26+%3D+%5Csum%5Climits_%7Bi%3D1%7D%5Em%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5E%7Bj%7D%29log%5Cfrac%7BP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5E%7Bj%2B1%7D%29%7D%7BP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5Ej%29%7D+%5Ctag%7B3%7D+%5C%5C+%26+%5Cleq+%5Csum%5Climits_%7Bi%3D1%7D%5Emlog%28%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5E%7Bj%7D%29%5Cfrac%7BP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5E%7Bj%2B1%7D%29%7D%7BP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5Ej%29%7D%29+%5Ctag%7B4%7D+%5C%5C+%26+%3D+%5Csum%5Climits_%7Bi%3D1%7D%5Emlog%28%5Csum%5Climits_%7Bz%5E%7B%28i%29%7D%7DP%28+z%5E%7B%28i%29%7D%7Cx%5E%7B%28i%29%7D%EF%BC%8C%5Ctheta%5E%7Bj%2B1%7D%29%29+%3D+0+%5Ctag%7B5%7D+%5Cend%7Balign%7D)

其中第（4）式用到了Jensen不等式，只不过和第二节的使用相反而已，第（5）式用到了概率分布累积为1的性质。

至此，我们得到了：![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%5E%7Bj%2B1%7D%29+-+%5Csum%5Climits_%7Bi%3D1%7D%5Em+logP%28x%5E%7B%28i%29%7D%7C%5Ctheta%5E%7Bj%7D%29+%5Cgeq+0) ，证明了EM算法的收敛性。

从上面的推导可以看出，EM 算法可以保证收敛到一个稳定点，但是却不能保证收敛到全局的极大值点，因此它是局部最优的算法，当然，如果我们的优化目标 ![[公式]](https://www.zhihu.com/equation?tex=L%28%CE%B8%2C%CE%B8%5Ej%29) 是凸的，则EM算法可以保证收敛到全局极大值，这点和梯度下降法这样的迭代算法相同。至此我们也回答了上面提到的第二个问题。

## 2.6. EM算法应用

如果我们从算法思想的角度来思考EM算法，我们可以发现我们的算法里已知的是观察数据，未知的是隐含数据和模型参数，在E步，我们所做的事情是固定模型参数的值，优化隐含数据的分布，而在M步，我们所做的事情是固定隐含数据分布，优化模型参数的值。EM的应用包括：

- 支持向量机的SMO算法
- 混合高斯模型
- K-means
- 隐马尔可夫模型

## 3. [EM算法案例-两硬币模型](https://link.zhihu.com/?target=http%3A//ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf)

假设有两枚硬币A、B，以相同的概率随机选择一个硬币，进行如下的掷硬币实验：共做 5 次实验，每次实验独立的掷十次，结果如图中 a 所示，例如某次实验产生了H、T、T、T、H、H、T、H、T、H (H代表正面朝上)。a 是在知道每次选择的是A还是B的情况下进行，b是在不知道选择的是A还是B的情况下进行，问如何估计两个硬币正面出现的概率？

![img](https://pic2.zhimg.com/v2-a5b47206d802b392e0e72a23c6b7bb95_b.jpg)

**CASE a**

已知每个实验选择的是硬币A 还是硬币 B，重点是如何计算输出的概率分布，这其实也是极大似然求导所得。
 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%5Cunderset%7B%5Ctheta+%7D%7Bargmax%7DlogP%28Y%7C%5Ctheta%29+%26%3D+log%28%28%5Ctheta_B%5E5%281-%5Ctheta_B%29%5E5%29+%28%5Ctheta_A%5E9%281-%5Ctheta_A%29%29%28%5Ctheta_A%5E8%281-%5Ctheta_A%29%5E2%29+%28%5Ctheta_B%5E4%281-%5Ctheta_B%29%5E6%29+%28%5Ctheta_A%5E7%281-%5Ctheta_A%29%5E3%29+%29+%5C%5C+%26%3D+log%5B%28%5Ctheta_A%5E%7B24%7D%281-%5Ctheta_A%29%5E6%29+%28%5Ctheta_B%5E9%281-%5Ctheta_B%29%5E%7B11%7D%29+%5D+%5Cend%7Balign%2A%7D)
上面这个式子求导之后发现，5 次实验中A正面向上的次数再除以总次数作为即为  ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%CE%B8_A) ，5次实验中B正面向上的次数再除以总次数作为即为 ，即:

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_A+%3D+%5Cfrac%7B24+%7D%7B24%2B6%7D+%3D+0.80+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_B+%3D+%5Cfrac%7B9%7D%7B+9+%2B+11%7D+%3D+0.45+%5C%5C)

**CASE b**

由于并不知道选择的是硬币 A 还是硬币 B，因此采用EM算法。

E步：初始化![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%CE%B8_A%5E%7B%280%29%7D+%3D+0.60)和 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+%CE%B8_B%5E%7B%280%29%7D+%3D+0.50) ，计算每个实验中选择的硬币是 A 和 B 的概率，例如第一个实验中选择 A 的概率为：

![[公式]](https://www.zhihu.com/equation?tex=P%28z%3DA%7Cy_1%2C+%5Ctheta%29+%3D+%5Cfrac+%7BP%28z%3DA%2C+y_1%7C%5Ctheta%29%7D%7BP%28z%3DA%2Cy_1%7C%5Ctheta%29+%2B+P%28z%3DB%2Cy_1%7C%5Ctheta%29%7D+%3D+%5Cfrac%7B%280.6%29%5E5%2A%280.4%29%5E5%7D%7B%280.6%29%5E5%2A%280.4%29%5E5%2B%280.5%29%5E%7B10%7D%7D+%3D+0.45+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=P%28z%3DB%7Cy_1%2C+%5Ctheta%29+%3D+1-+P%28z%3DA%7Cy_1%2C+%5Ctheta%29+%3D+0.55+%5C%5C)

计算出每个实验为硬币 A 和硬币 B 的概率，然后进行加权求和。

**M步**：求出似然函数下界 ![[公式]](https://www.zhihu.com/equation?tex=+Q%28%5Ctheta%2C+%5Ctheta%5Ei%29)， ![[公式]](https://www.zhihu.com/equation?tex=y_j)代表第 ![[公式]](https://www.zhihu.com/equation?tex=j) 次实验正面朝上的个数，![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_j) 代表第 ![[公式]](https://www.zhihu.com/equation?tex=j) 次实验选择硬币 A 的概率，![[公式]](https://www.zhihu.com/equation?tex=1-%5Cmu_j) 代表第 ![[公式]](https://www.zhihu.com/equation?tex=j) 次实验选择硬币B的概率 。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+Q%28%5Ctheta%2C+%5Ctheta%5Ei%29+%26%3D+%5Csum_%7Bj%3D1%7D%5E5%5Csum_%7Bz%7D+P%28z%7Cy_j%2C+%5Ctheta%5Ei%29logP%28y_j%2C+z%7C%5Ctheta%29%5C%5C%26%3D%5Csum_%7Bj%3D1%7D%5E5+%5Cmu_jlog%28%5Ctheta_A%5E%7By_j%7D%281-%5Ctheta_A%29%5E%7B10-y_j%7D%29+%2B+%281-%5Cmu_j%29log%28%5Ctheta_B%5E%7By_j%7D%281-%5Ctheta_B%29%5E%7B10-y_j%7D%29+%5Cend%7Balign%2A%7D)

针对L函数求导来对参数求导，例如对 ![[公式]](https://www.zhihu.com/equation?tex=%CE%B8_A)求导：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%5Cfrac%7B%5Cpartial+Q%7D%7B%5Cpartial+%5Ctheta_A%7D+%26%3D+%5Cmu_1%28%5Cfrac%7By_1%7D%7B%5Ctheta_A%7D-%5Cfrac%7B10-y_1%7D%7B1-%5Ctheta_A%7D%29+%2B+%5Ccdot+%5Ccdot+%5Ccdot+%2B+%5Cmu_5%28%5Cfrac%7By_5%7D%7B%5Ctheta_A%7D-%5Cfrac%7B10-y_5%7D%7B1-%5Ctheta_A%7D%29+%3D+%5Cmu_1%28%5Cfrac%7By_1+-+10%5Ctheta_A%7D+%7B%5Ctheta_A%281-%5Ctheta_A%29%7D%29+%2B+%5Ccdot+%5Ccdot+%5Ccdot+%2B%5Cmu_5%28%5Cfrac%7By_5+-+10%5Ctheta_A%7D+%7B%5Ctheta_A%281-%5Ctheta_A%29%7D%29+%5C%5C+%26%3D+%5Cfrac%7B%5Csum_%7Bj%3D1%7D%5E5+%5Cmu_jy_j+-+%5Csum_%7Bj%3D1%7D%5E510%5Cmu_j%5Ctheta_A%7D+%7B%5Ctheta_A%281-%5Ctheta_A%29%7D+%5Cend%7Balign%2A%7D+%5C%5C)

求导等于 0 之后就可得到图中的第一次迭代之后的参数值:

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_A%5E%7B%281%29%7D+%3D+0.71+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_B%5E%7B%281%29%7D+%3D+0.58+%5C%5C)

当然，基于Case a 我们也可以用一种更简单的方法求得：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_A%5E%7B%281%29%7D+%3D+%5Cfrac%7B21.3%7D%7B21.3%2B8.6%7D+%3D+0.71+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_B%5E%7B%281%29%7D+%3D+%5Cfrac%7B11.7%7D%7B+11.7+%2B+8.4%7D+%3D+0.58+%5C%5C)

**第二轮迭代**：基于第一轮EM计算好的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_A%5E%7B%281%29%7D%2C+%5Chat%7B%5Ctheta%7D_B%5E%7B%281%29%7D) , 进行第二轮 EM，计算每个实验中选择的硬币是 A 和 B 的概率（E步），然后在计算M步，如此继续迭代......迭代十步之后 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D_A%5E%7B%2810%29%7D+%3D+0.8%2C+%5Chat%7B%5Ctheta%7D_B%5E%7B%2810%29%7D+%3D+0.52)



**引用文献：**

1.[《从最大似然到EM算法浅解》](https://link.zhihu.com/?target=http%3A//blog.csdn.net/zouxy09/article/details/8537620)

\2. Andrew Ng 《[Mixtures of Gaussians and the EM algorithm](https://link.zhihu.com/?target=http%3A//cs229.stanford.edu/notes/cs229-notes7b.pdf)》

\3. 《[What is the expectation maximization algorithm?](https://link.zhihu.com/?target=http%3A//ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf)》
