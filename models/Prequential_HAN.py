import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import nltk
import h5py
from keras import backend as K
from keras.models import Model
from keras import initializers
from tensorflow.keras.layers import Layer
from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, LSTM, Bidirectional, TimeDistributed
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical

from nltk import tokenize
from nltk.tokenize import sent_tokenize
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

nltk.download('punkt')

maxlen = 20
max_sentences = 50
max_words = 1000000
embedding_dim = 100
reviews = []
labels = []
texts = []
valid_reviews = []
valid_labels = []
valid_texts = []
test_reviews = []
test_labels = []
test_texts = []
glove_dir = "./glove.6B"
embeddings_index = {}

input_evd = pd.read_pickle('data/train.pkl')
val_evd = pd.read_pickle('data/valid.pkl')
test_evd = pd.read_pickle('data/test.pkl')

input_evd = input_evd.sort_values(['date'])
val_evd = val_evd.sort_values(['date'])
test_evd = test_evd.sort_values(['date'])

total_evd = pd.concat([input_evd, val_evd, test_evd])
total_evd = total_evd.sort_values(['date'])


import datetime
from dateutil.relativedelta import relativedelta

# class defining the custom attention layer
class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name="W")
        self.b = K.variable(self.init((self.attention_dim,)), name="b")
        self.u = K.variable(self.init((self.attention_dim, 1)), name="u")
        self.train_wei = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(HierarchicalAttentionNetwork, self).get_config()
        config.update({"supports_masking" : self.supports_masking,
                       "attention_dim" : self.attention_dim,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


# build reviews and labels set
total_reviews = []
total_texts = []
total_labels = []
total_dates = []

for idx in range(total_evd.shape[0]):
    claim = total_evd.iloc[idx]['text'].strip()
    evidence = total_evd.iloc[idx]['evidence'].strip()
    text = claim + ' ' + evidence
    text = text.replace(u'\xa0', u' ')
    total_texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    total_reviews.append(sentences)
    total_labels.append(total_evd.iloc[idx]['rating'])
    temp_date = total_evd.iloc[idx]['date'][:10]
    total_dates.append(datetime.date(int(temp_date[:4]), int(temp_date[5:7]), int(temp_date[8:])))

tokenizer = Tokenizer(num_words=max_words, filters='!"“”#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ', char_level=False, oov_token=None)
tokenizer.fit_on_texts(total_texts)
total_data = np.zeros((len(total_texts), max_sentences, maxlen), dtype='int32')

for i, sentences in enumerate(total_reviews):
    for j, sent in enumerate(sentences):
        if j < max_sentences:
            sent = sent.replace(u'\xa0', u' ')
            wordTokens = text_to_word_sequence(sent,
                                               filters = '!"“”#$%&()*+,-./\:;<=>?@[\\]^_`{|}~\t\n',
                                               lower = True,
                                               split = " ")
            k = 0
            for _, word in enumerate(wordTokens):
                if word == 'healthpocket':
                     k = k+1
                elif k < maxlen and tokenizer.word_index[word] < max_words:
                    total_data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1


word_index = tokenizer.word_index
total_labels = to_categorical(np.asarray(total_labels))
timespan = relativedelta(months=6)
starttime = datetime.date(2010, 1, 1)
endtime = datetime.date(2021, 7, 30)


# build roll-forward (prequential) train, valid and test sets with respect to timeline
trainsetsX = []
trainsetsY = []
testsetsX = []
testsetsY = []

first_train_index = 0
for i in range(total_data.shape[0]):
    if total_dates[i] < starttime:
        continue
    else:
      first_train_index = i
      break
first_train_X = total_data[:first_train_index+1]
first_train_Y = total_labels[:first_train_index+1]

trainsetsX.append(first_train_X)
trainsetsY.append(first_train_Y)
first_test_time = starttime + timespan

first_test_index = 0
for i in range(total_data.shape[0]):
    if total_dates[i] < first_test_time:
        continue
    else:
      first_test_index = i
      break
first_test_X = total_data[first_train_index+1:first_test_index+1]
first_test_Y = total_labels[first_train_index+1:first_test_index+1]

testsetsX.append(first_test_X)
testsetsY.append(first_test_Y)

cur1 = first_test_time
cur2 = cur1 + timespan
while cur2 < endtime:
    trainind = 0
    testind = 0
    for i in range(total_data.shape[0]):
        if total_dates[i] < cur1:
            continue
        else:
            trainind = i
            break
    for j in range(trainind, total_data.shape[0]):
        if total_dates[j] < cur2:
            continue
        else:
            testind = j
            break
    trX = total_data[:trainind+1]
    trY = total_labels[:trainind+1]
    tstX = total_data[trainind+1:testind+1]
    tstY = total_labels[trainind+1:testind+1]
    trainsetsX.append(trX)
    trainsetsY.append(trY)
    testsetsX.append(tstX)
    testsetsY.append(tstY)
    cur1 = cur1 + timespan
    cur2 = cur2 + timespan


# use Glove for embeddings
f = open('glove.6B.100d.txt','r+', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# prequential train and evaluation
train_val_ratio = 0.8
for j in range(len(trainsetsX)):
    checkpoint_filepath = 'checkpoint.hdf5'
    mc = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_acc', mode='max', verbose=1)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)

    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=True)
    sentence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    lstm_word = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    attn_word = HierarchicalAttentionNetwork(100)(lstm_word)
    sentenceEncoder = Model(sentence_input, attn_word)
    review_input = Input(shape=(max_sentences, maxlen), dtype='int32')
    review_encoder = TimeDistributed(sentenceEncoder)(review_input)
    lstm_sentence = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
    attn_sentence = HierarchicalAttentionNetwork(100)(lstm_sentence)
    preds = Dense(3, activation='softmax')(attn_sentence)

    model = Model(review_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
    print("model fitting - Hierachical attention network")
    print("This is the %d-th rolling window training process" % j)
    print("--------------------------------------------------------------\n")
    indices = np.arange(trainsetsX[j].shape[0])
    np.random.shuffle(indices)

    # use corresponding prequential train and valid set
    x_train = trainsetsX[j][indices][:int(train_val_ratio * len(indices))]
    y_train = np.array(trainsetsY[j])[indices][:int(train_val_ratio * len(indices))]
    x_val = trainsetsX[j][indices][int(train_val_ratio * len(indices)):]
    y_val = np.array(trainsetsY[j])[indices][int(train_val_ratio * len(indices)):]

    ts_indices = np.arange(testsetsX[j].shape[0])
    np.random.shuffle(ts_indices)
    x_test = np.array(testsetsX[j])[ts_indices]
    y_test = np.array(testsetsY[j])[ts_indices]
    print("%d-th window: Start training for model with crossentropy loss" % (j))
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, callbacks = [es, mc])
    print("Loading best model for categorical crossentropy ...")
    model.load_weights(checkpoint_filepath)

    print("Evaluate on test data based on best model")
    results = model.evaluate(x_test, y_test, batch_size=64)
    print(results)
    print("Calculate the macro F1-score")
