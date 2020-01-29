import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
# color = sns.color_palette()
import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
# offline.init_notebook_mode()
#import cufflinks and offline mode
import cufflinks as cf
# cf.go_offline()

# Venn diagram
from matplotlib_venn import venn2
import re
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
eng_stopwords = stopwords.words('english')
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


import os
# print(os.path.abspath('.'))
print(os.listdir('../Google-QUEST-Q&A-Labeling/input'))

print('Reading data...')
train_data = pd.read_csv('../Google-QUEST-Q&A-Labeling/input/train.csv')
test_data = pd.read_csv('../Google-QUEST-Q&A-Labeling/input/test.csv')
sample_submission = pd.read_csv('../Google-QUEST-Q&A-Labeling/input/sample_submission.csv')
print('Reading data completed')


# print('Size of train_data',train_data.shape)
# print('Size of test_data',test_data.shape)
# print('Size of sample_submission',sample_submission.shape)

# print(train_data.head())
# print(train_data.columns)
# print(test_data.head())
# print(test_data.columns)
# print(sample_submission.head())
# targets = list(sample_submission.columns[1:])
# print(targets)

# print(train_data[targets].describe())


# total = train_data.isnull().sum().sort_values(ascending=False)
# percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending=False)
# missing_train_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
# print(missing_train_data.head())

# total = test_data.isnull().sum().sort_values(ascending = False)
# percent = (test_data.isnull().sum()/test_data.isnull().count()*100).sort_values(ascending = False)
# missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_test_data.head())

temp = train_data['host'].value_counts()
df = pd.DataFrame({'labels':temp.index,'values':temp.values})
df.iplot(kind='pie',labels='labels',values='values',title='Distribution of hosts in Training data')