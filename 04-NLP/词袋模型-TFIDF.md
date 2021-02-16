## 引言
- 在数据挖掘算法中，通常我们可以将结构化的数据很好的通过向量化的形式变现出来。

| 年龄 | 年收入  |
| ---- | ------- |
| 18   | 20000   |
| 35   | 1000000 |

- 每一个属性特征都能进行量化，他们都是独立包含信息的。对与第一个人，我们可以直接用[18,20000]来表示他的属性内涵。而词语，句子该如何表示呢？
- 本文就介绍几种简单的文本表示方式：词集模型/词袋模型/TF-IDF/n-grams



### 词集和词袋模型

- 词集模型(Set of Word,SOW):单词构成集合，每个单词出现则为1，不出现则为0。
- 词袋模型(Bag of Word,BOW):统计每个词的频率，每个词以词频表示。
- 举个栗子
  - 句子一：I love her,but she don't love me.
  - 句子二：I love her,and she love me too.
  - 形成一个词表{'I':2,'love':4,'her':2,'but':1,'she':2,'don't':1,'me':2,'and':1,'too':1}
  - 词集模型：句子一[1,1,1,1,1,1,1,0,0]
  - 词袋模型：句子二[2,4,1,0,1,0,1,0,1]

```python
# 使用sklearn的CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                encoding='utf-8', input='content',
                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                ngram_range=(1, 1), preprocessor=None, stop_words=None,
                strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None, vocabulary=None)
corpus = ['I love her,but she don\'t love me.',
            'I love her,and she love me too.']
X = vec.fit_transform(corpus)
print(X)
>>>
  (0, 4)        2
  (0, 3)        1
  (0, 1)        1
  (0, 6)        1
  (0, 2)        1
  (0, 5)        1
  (1, 4)        2
  (1, 3)        1
  (1, 6)        1
  (1, 5)        1
  (1, 0)        1
  (1, 7)        1
# 查看去重后的单词～
names = vec.get_feature_names()
print(names)
>>>
['and', 'but', 'don', 'her', 'love', 'me', 'she', 'too']
# 把这两句话向量化！
array = X.toarray()
print(array)
>>>
[[0 1 1 1 2 1 1 0]
 [1 0 0 1 2 1 1 1]]
```

### CountVectorizer相关详解

- Analyzer:string,('word','char','char_wb')
- Binary:如果是True，那么让数量为0的词有一个默认数值1。
- Decode_errors:strict是默认的，另外还有ignore，replace。
- encoding：给分析器的解码方式默认是utf-8
- Lowercase:默认值为True，在tokenizing前变成小写。
- max_df:可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。如果参数中已经给定了vocabulary，则这个参数无效。
- Min_df:和max_df类似。
- Max_feature:只考虑前n频率的特征。
- Ngram_range:(1,1)代表unigrams即单字符，(1,2)代表unigrams和bigrams,(2,2)代表bigrams。
- stop_words:可以是‘English’，list，None。
  - English：一个内置的停用词库被调用
  - list：把自己想要的停用词填进去，这些词不会再出现在result token里面



### TF-IDF

- 我们很直观的可以想到，如果某个词重要，那么他出现的次数肯定是会比较多的。出现次数我们一般叫做“词频”。但是同时会出现“我”，“是”这些常见的词或字，出现频率很高，而代表一段话的意义很低，我们可以通过停用词的筛选过滤掉一部分，但也没办法做到完全过滤。
- 那么，引申出了另一个概念“词频-逆文档频”也就是tf-idf。

- TF：$\frac{count(w_{i})}{D_{i}}$
  - $w_{i}$:这个是代表第i个词，整个分子代表着这个词出现的次数。
  - $D_{i}$:代表整个文档的词语数量。
- IDF：$log\frac{N}{1+\sum^{N}_{i=1}I(w,D_{i})}$
  - N:所有文档的总数
  - 1:防止分母为0
  - $I(w,D_{i})$:表示文档D中是否包含词w，包含就加一



### 使用Gensim

- 对语料做一个简单的分词处理

```python
from pprint import pprint
corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document'
]

word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))
pprint(word_list)
>>>
[['this', 'is', 'the', 'first', 'document'],
 ['this', 'is', 'the', 'second', 'second', 'document'],
 ['and', 'the', 'third', 'one'],
 ['is', 'this', 'the', 'first', 'document']]
```

- 获取每个词的id和词频

```python
from gensim import corpora

dictionary = corpora.Dictionary(word_list)
new_corpus = [dictionary.doc2bow(text) for text in word_list]
print(new_corpus)
>>>
# 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
 [(0, 1), (2, 1), (3, 1), (4, 1), (5, 2)],
 [(3, 1), (6, 1), (7, 1), (8, 1)],
 [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]]

 # 通过下面的方法可以看到语料库中每个词对应的id
 print(dictionary.token2id)
>>>
{'and': 6,
 'document': 0,
 'first': 1,
 'is': 2,
 'one': 7,
 'second': 5,
 'the': 3,
 'third': 8,
 'this': 4}
```

- 训练gensim模型

```python
from gensim import models
tfidf = models.TfidfModel(new_corpus)
tfidf.save("my_model.tfidf")
# 训练模型并保存
from gensim import models
tfidf = models.TfidfModel(new_corpus)
tfidf.save("my_model.tfidf")

# 载入模型
tfidf = models.TfidfModel.load("my_model.tfidf")

# 使用这个训练好的模型得到单词的tfidf值
tfidf_vec = []
for i in range(len(corpus)):
    string = corpus[i]
    string_bow = dictionary.doc2bow(string.lower().split())
    string_tfidf = tfidf[string_bow]
    tfidf_vec.append(string_tfidf)
print(tfidf_vec)

>>>
[[(0, 0.33699829595119235),
  (1, 0.8119707171924228),
  (2, 0.33699829595119235),
  (4, 0.33699829595119235)],
 [(0, 0.10212329019650272),
  (2, 0.10212329019650272),
  (4, 0.10212329019650272),
  (5, 0.9842319344536239)],
 [(6, 0.5773502691896258), (7, 0.5773502691896258), (8, 0.5773502691896258)],
 [(0, 0.33699829595119235),
  (1, 0.8119707171924228),
  (2, 0.33699829595119235),
  (4, 0.33699829595119235)]]

# 我们随便拿几个单词来测试
string = 'the i first second name'
string_bow = dictionary.doc2bow(string.lower().split())
string_tfidf = tfidf[string_bow]
print(string_tfidf)

>>>
# 自动去除了停用词
[(1, 0.4472135954999579), (5, 0.8944271909999159)]
```



### 使用Sklearn

```python
corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document'
]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)

# 得到语料库所有不重复的词
print(tfidf_vec.get_feature_names())
>>>['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
# 得到每个单词对应的id值
print(tfidf_vec.vocabulary_)
>>>{'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}
# 得到每个句子所对应的向量
# 向量里数字的顺序是按照词语的id顺序来的
print(tfidf_matrix.toarray())
>>>
array([[0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
        0.        , 0.35872874, 0.        , 0.43877674],
       [0.        , 0.27230147, 0.        , 0.27230147, 0.        ,
        0.85322574, 0.22262429, 0.        , 0.27230147],
       [0.55280532, 0.        , 0.        , 0.        , 0.55280532,
        0.        , 0.28847675, 0.55280532, 0.        ],
       [0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
        0.        , 0.35872874, 0.        , 0.43877674]])
```



### 参考

- [https://blog.csdn.net/qq_27586341/article/details/90286751#1.%20%E8%AF%8D%E9%9B%86%E6%A8%A1%E5%9E%8B](https://blog.csdn.net/qq_27586341/article/details/90286751#1. 词集模型)
- https://zhuanlan.zhihu.com/p/53302305
- https://www.jianshu.com/p/f3b92124cd2b