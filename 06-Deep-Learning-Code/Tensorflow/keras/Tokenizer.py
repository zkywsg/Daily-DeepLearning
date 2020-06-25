from keras.preprocessing import sequence,text

token = text.Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                       lower=True,split=' ')
max_len = 3

token.fit_on_texts(['I am a poor guy','You are a rich man'])
xtoken = token.texts_to_sequences(['I am a poor guy','You are a rich man'])
print(xtoken)
# [[2, 3, 1, 4, 5], [6, 7, 1, 8, 9]]

# zero pad the swquences
xtoken_pad = sequence.pad_sequences(xtoken,maxlen=max_len)
print(xtoken_pad)
# [[1 4 5]
#  [1 8 9]]

word_index = token.word_index
print(word_index)
# {'a': 1, 'i': 2, 'am': 3, 'poor': 4, 'guy': 5, 'you': 6, 'are': 7, 'rich': 8, 'man': 9}

