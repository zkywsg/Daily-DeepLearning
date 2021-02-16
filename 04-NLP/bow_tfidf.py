# coding:utf-8

# @Author: zkywsg
# @Date: 2020-07-27

# from sklearn.feature_extraction.text import CountVectorizer
# vec = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#                 encoding='utf-8', input='content',
#                 lowercase=True, max_df=1.0, max_features=None, min_df=1,
#                 ngram_range=(1, 1), preprocessor=None, stop_words=None,
#                 strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#                 tokenizer=None, vocabulary=None)
# corpus = ['I love her,but she don\'t love me.',
#             'I love her,and she love me too.']
# X = vec.fit_transform(corpus)
# # print(X)

# names = vec.get_feature_names()
# # print(names)

# array = X.toarray()
# print(array)
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
# pprint(word_list)

from gensim import corpora
dictionary = corpora.Dictionary(word_list)
new_corpus = [dictionary.doc2bow(text) for text in word_list]
pprint(new_corpus)
# pprint(dictionary.token2id)

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
# pprint(tfidf_vec)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)

# 得到语料库所有不重复的词
pprint(tfidf_vec.get_feature_names())

# 得到每个单词对应的id值
print(tfidf_vec.vocabulary_)

# 得到每个句子所对应的向量
# 向量里数字的顺序是按照词语的id顺序来的
pprint(tfidf_matrix.toarray())