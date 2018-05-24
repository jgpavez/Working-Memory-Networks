''' 
    Relational network for BABI, based on
    https://github.com/Alan-Lee123/relation-network
    
'''

from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout, RepeatVector, Lambda, Permute, Activation, Masking, Layer
from keras.layers import recurrent, Input, merge, TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_accuracy
import keras.backend as K

from keras import initializations, regularizers, constraints

from functools import reduce
import tarfile
import numpy as np
#np.random.seed(1337)  # for reproducibility

import re
import pdb
import time

from itertools import izip_longest

from os import listdir
from os.path import isfile, join

import sys
sys.setrecursionlimit(10000)

from utils import vectorize_facts, get_stories

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

class SequenceEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, position_encoding=False, **kwargs):
        self.position_encoding = position_encoding
        self.zeros_vector =  K.reshape(K.zeros(shape=(output_dim,)),(1,output_dim))
        super(SequenceEmbedding, self).__init__(input_dim, output_dim, **kwargs)
 
    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        W_ = K.concatenate([self.zeros_vector, W], axis=0)
        out = K.gather(W_, x)
        return out


tar = tarfile.open('babi-tasks-v1-2.tar.gz')

# Reading all file names
mypath = 'tasks_1-20_v1-2/en'
challenge_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
challenge_files = ['tasks_1-20_v1-2/en-10k/' + f.replace('train', '{}') for f 
                   in challenge_files if 'train.txt' == f[-9:]]

# Read all files
train_facts_split = []
test_facts_split = []
train_facts = []
test_facts = []
for challenge in challenge_files:
    train_facts_split.append(get_stories(tar.extractfile(challenge.format('train'))))
    test_facts_split.append(get_stories(tar.extractfile(challenge.format('test'))))
    train_facts += train_facts_split[-1]
    test_facts += test_facts_split[-1]

test_facts = np.array(test_facts)
train_facts = np.array(train_facts)
test_facts = list(test_facts[np.random.choice(len(test_facts), len(test_facts), replace=False)])
train_facts = list(train_facts[np.random.choice(len(train_facts), len(train_facts), replace=False)])

# Parameters deifinition
EMBED_HIDDEN_SIZE = 30
MLP_unit = 128
LSTM_unit = 30
enable_time = True

print('Processing input data')

train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in train_facts]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in test_facts]

facts_maxlen = max(map(len, (x for h,_,_ in train_facts + test_facts for x in h)))
if enable_time:
    facts_maxlen += 1

story_maxlen = max(map(len, (x for x, _, _ in train_facts + test_facts)))
query_maxlen = max(map(len, (x for _, x, _ in train_facts + test_facts)))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Num facts:', facts_maxlen, 'facts')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))

print('Vectorizing the word sequences...')

story_maxlen = 20

if enable_time:
    vocab_size += story_maxlen

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_facts(train_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                               enable_time=enable_time)
inputs_test, queries_test, answers_test = vectorize_facts(test_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                         enable_time=enable_time)


fact_input = Input(shape=(story_maxlen, facts_maxlen, ), dtype='int32', name='facts_input')
question_input = Input(shape=(query_maxlen, ), dtype='int32', name='query_input')

question_layer = SequenceEmbedding(input_dim=vocab_size-1,
                               output_dim=EMBED_HIDDEN_SIZE,
                               input_length=query_maxlen, init='normal')

question_encoder = question_layer(question_input)

question_encoder = LSTM(LSTM_unit, return_sequences=False)(question_encoder)


layer_encoder = SequenceEmbedding(input_dim=vocab_size-1,
                       output_dim=EMBED_HIDDEN_SIZE,
                       input_length=story_maxlen, init='normal')

input_encoder = layer_encoder(fact_input)
input_encoder = Lambda(lambda x: x,
                        output_shape=lambda shape: (None, story_maxlen, facts_maxlen, 
                                                    EMBED_HIDDEN_SIZE,))(input_encoder)

input_encoder = TimeDistributed(LSTM(LSTM_unit, return_sequences=False))(input_encoder)

objects = []
for k in range(story_maxlen):
    fact_object = Lambda(lambda x: x[:,k,:], 
                         output_shape=(LSTM_unit,))(input_encoder)
    objects.append(fact_object)

relations = []
for fact_object_1 in objects:
    for fact_object_2 in objects:
        relations.append(merge([fact_object_1, fact_object_2, question_encoder], mode='concat',
                              output_shape=(None, LSTM_unit * 3,)))

from keras.layers.normalization import BatchNormalization


def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f


def get_MLP(n):
    r = []
    for k in range(n):
        s = stack_layer([
            Dense(MLP_unit, input_shape=(LSTM_unit * 3,)),
            #BatchNormalization(mode=2),
            Activation('relu')
        ])
        r.append(s)
    return stack_layer(r)

g_MLP = get_MLP(4)
mid_relations = []
for r in relations:
    mid_relations.append(g_MLP(r))
combined_relation = merge(mid_relations, mode='sum')

def bn_dense(x, units):
    y = Dense(units)(x)
    #y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    return y

rn = bn_dense(combined_relation, 256)
rn = bn_dense(rn, 512)
rn = bn_dense(rn, 159)
response = Dense(vocab_size, init='normal', activation='softmax')(combined_relation)

model = Model(input=[fact_input, question_input], output=[response])

#model.summary()

adam = Adam(lr=2.5e-4)
            
print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[categorical_accuracy])
print('Compilation done...')

# Train / Validation Split
choices = np.random.choice(len(train_facts), len(train_facts), replace=False)
train_facts = np.array(train_facts)
valid_facts = train_facts[choices[-int(len(train_facts)*0.1):]]
train_facts = train_facts[choices[:int(len(train_facts)*0.9)]]

inputs_valid, queries_valid, answers_valid = vectorize_facts(valid_facts, word_idx, 
                                                             story_maxlen, query_maxlen, facts_maxlen,
                                                             enable_time=enable_time)

BATCH_SIZE = 32

show_batch_interval = 1000

EPOCHS = 5000
N_BATCHS = len(train_facts) // BATCH_SIZE
EARLY_STOP_MAX = 4

save_hist = []
save_hist.append(0.)

for k in xrange(EPOCHS):
    for b,batch in enumerate(grouper(train_facts, BATCH_SIZE, fillvalue=train_facts[-1])):
        inputs_train, queries_train, answers_train = vectorize_facts(batch, word_idx, 
                                                                     story_maxlen, query_maxlen, facts_maxlen,
                                                                     enable_time=enable_time)
        start = time.time()
        loss = model.train_on_batch([inputs_train, queries_train], answers_train)
        end = time.time()
        if b % show_batch_interval == 0:
            print('Epoch: {0}, Batch: {1}, loss: {2} - acc: {3}, time: {4:.3f}s'.format(k, 
                                                                        b, float(loss[0]), float(loss[1]),
                                                                        (end-start)*1000.))

    losses = model.evaluate([inputs_valid, queries_valid], answers_valid, batch_size=BATCH_SIZE, 
                            verbose=0)
    print('Epoch {0}, valid loss / valid accuracy: {1} / {2}'.
           format(k, losses[0], losses[1]))
    
    if max(save_hist) < losses[1]:
        model.save_weights('models/weights_rn.hdf5', overwrite=True)
    save_hist.append(losses[1])
    
    if max(save_hist) > losses[1]:
        early_stop += 1
    else:
        early_stop = 0

print('Total Model Accuracy: ')
loss, acc = model.evaluate([inputs_test, queries_test], answers_test)
print('Loss: {0}, Acc: {1}'.format(loss, acc))
print('Per-Task Accuracy: ')
passed = 0
for k, challenge in enumerate(challenge_files):
    test_fact = test_facts_split[k]
    print(challenge)
    inputs_test, queries_test, answers_test = vectorize_facts(test_fact, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                         enable_time=enable_time)
    loss, acc = model.evaluate([inputs_test, queries_test], answers_test)
    print('\n Loss: {0}, Acc: {1}, Pass: {2} '.format(loss, acc, acc >= 0.95))
    passed += acc >= 0.95
print ('Passed: {0}'.format(passed))
