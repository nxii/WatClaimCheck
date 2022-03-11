import numpy as np
import pandas as pd
import tensorflow as tf
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

nltk.download('punkt')
tf.executing_eagerly()

# hyper-parameters settings
maxlen = 20
max_sentences = 30
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

# use data generation script to get train.pkl, valid.pkl and test.pkl
input_evd = pd.read_pickle('train.pkl')
val_evd = pd.read_pickle('valid.pkl')
test_evd = pd.read_pickle('test.pkl')

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


# build texts, sentences and reviews on train, valid and test data
for idx in range(input_evd.shape[0]):
    claim = input_evd.iloc[idx]['text'].strip()
    text = claim
    evidence = input_evd.iloc[idx]['evidence_sents']
    for elem in evidence:
      text = text + ' ' + elem.strip()
    text = text.replace(u'\xa0', u' ')
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)
    labels.append(input_evd.iloc[idx]['rating'])


tokenizer = Tokenizer(num_words=max_words, filters='!"“”#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ', char_level=False, oov_token=None)
tokenizer.fit_on_texts(texts)


for idx in range(val_evd.shape[0]):
    claim = val_evd.iloc[idx]['text'].strip()
    text = claim
    evidence = val_evd.iloc[idx]['evidence_sents']
    for elem in evidence:
      text = text + ' ' + elem.strip()
    text = text.replace(u'\xa0', u' ')
    valid_texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    valid_reviews.append(sentences)
    valid_labels.append(val_evd.iloc[idx]['rating'])

tokenizer.fit_on_texts(valid_texts)


for idx in range(test_evd.shape[0]):
    claim = test_evd.iloc[idx]['text'].strip()
    text = claim
    evidence = test_evd.iloc[idx]['evidence_sents']
    for elem in evidence:
      text = text + ' ' + elem.strip()
    text = text.replace(u'\xa0', u' ')
    test_texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    test_reviews.append(sentences)
    test_labels.append(test_evd.iloc[idx]['rating'])

tokenizer.fit_on_texts(test_texts)


# aggregate data, build tensor
data = np.zeros((len(texts), max_sentences, maxlen), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < max_sentences:
            sent = sent.replace(u'\xa0', u' ')
            wordTokens = text_to_word_sequence(sent,
                                               filters = '!"“”#$%&()*+,-./\:;<=>?@[\\]^_`{|}~\t\n',
                                               lower = True,
                                               split = " ")
            k = 0
            for _, word in enumerate(wordTokens):
                if word in ['healthpocket', '…\u2009the'] or (i == 2217 and k == 0):
                    k = k+1 
                elif k < maxlen and tokenizer.word_index[word] < max_words:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1


vl_data = np.zeros((len(valid_texts), max_sentences, maxlen), dtype='int32')

for i, sentences in enumerate(valid_reviews):
    for j, sent in enumerate(sentences):
        if j < max_sentences:
            sent = sent.replace(u'\xa0', u' ')
            wordTokens = text_to_word_sequence(sent,
                                               filters = '!"“”#$%&()*+,-./\:;<=>?@[\\]^_`{|}~\t\n',
                                               lower = True,
                                               split = " ")
            k = 0
            for _, word in enumerate(wordTokens):
                if k < maxlen and tokenizer.word_index[word] < max_words:
                    vl_data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1


ts_data = np.zeros((len(test_texts), max_sentences, maxlen), dtype='int32')

for i, sentences in enumerate(test_reviews):
    for j, sent in enumerate(sentences):
        if j < max_sentences:
            sent = sent.replace(u'\xa0', u' ')
            wordTokens = text_to_word_sequence(sent,
                                               filters = '!"“”#$%&()*+,-./\:;<=>?@[\\]^_`{|}~\t\n',
                                               lower = True,
                                               split = " ")
            k = 0
            for _, word in enumerate(wordTokens):
                if k < maxlen and tokenizer.word_index[word] < max_words:
                    ts_data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1


# align labels with text data
word_index = tokenizer.word_index
labels = to_categorical(np.asarray(labels))
valid_labels = to_categorical(np.asarray(valid_labels))
test_labels = to_categorical(np.asarray(test_labels))

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data
y_train = labels

vl_indices = np.arange(vl_data.shape[0])
np.random.shuffle(vl_indices)
vl_data = vl_data[vl_indices]
valid_labels = valid_labels[vl_indices]
x_val = vl_data
y_val = valid_labels

ts_indices = np.arange(ts_data.shape[0])
np.random.shuffle(ts_indices)
ts_data = ts_data[ts_indices]
test_labels = test_labels[ts_indices]
x_test = ts_data
y_test = test_labels


# use glove for embeddings
f = open('glove.6B.100d.txt','r+', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# build Hierachical Attention network
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



# use patience and early stopping during training
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

checkpoint_filepath = 'checkpoint.hdf5'
mc = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, 
                     monitor='val_acc', mode='max', verbose=1)
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)

model = Model(review_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, callbacks = [es, mc])
model.load_weights(checkpoint_filepath)
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=64)
print("test loss, test acc:", results)
