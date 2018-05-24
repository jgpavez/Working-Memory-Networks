'''
  Implementation of a Working Memory Network for the bAbI QA dataset.
  authors: jgpavez
'''

from __future__ import print_function

import numpy as np
import random
import sys
import theano
from functools import reduce
import time
import re
import copy
from itertools import izip_longest
import tarfile

from os import listdir
from os.path import isfile, join
seed = 1234

if len(sys.argv) > 6:
    seed = int(sys.argv[6])

np.random.seed(seed)
random.seed(seed)

from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout, RepeatVector, Lambda, Permute, Activation, Reshape
from keras.layers import Input, merge, TimeDistributed
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.layers.core import Layer
from keras.activations import softmax, tanh, sigmoid, hard_sigmoid, relu


from keras.preprocessing.sequence import pad_sequences
from keras.metrics import categorical_accuracy
import keras.backend as K
from keras.regularizers import l2, activity_l2

from s_model import sModel
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

def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f

# Model & Training parameters
EMBED_HIDDEN_SIZE = 30
LSTM_HIDDEN_UNITS = 30
G_HIDDEN_UNITS = 128
MXLEN = 30
RUN_DEC = 0
EPOCHS = 400
LEARNING_RATE = 0.001
enable_time = True

if len(sys.argv) > 1:
    EMBED_HIDDEN_SIZE = int(sys.argv[1])
if len(sys.argv) > 2:
    LSTM_HIDDEN_UNITS = int(sys.argv[2])
if len(sys.argv) > 3:
    MXLEN = int(sys.argv[3])
if len(sys.argv) > 4:
    RUN_DEC = int(sys.argv[4])
if len(sys.argv) > 5:
    LEARNING_RATE = float(sys.argv[5])

if RUN_DEC > 0:
    EPOCHS = 20
    
print(EMBED_HIDDEN_SIZE, LSTM_HIDDEN_UNITS, MXLEN, RUN_DEC, LEARNING_RATE, seed)
save_str = '{0}_{1}_{2}_{3}_{4}'.format(EMBED_HIDDEN_SIZE, LSTM_HIDDEN_UNITS, MXLEN, RUN_DEC, seed)

print ('Reading babi files')
tar = tarfile.open('data/babi/babi-tasks-v1-2.tar.gz')

mypath = 'data/babi/tasks_1-20_v1-2/en-10k'
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

print('Processing input data')
train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in train_facts]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in test_facts]

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1

facts_maxlen = max(map(len, (x for h,_,_ in train_facts + test_facts for x in h)))
if enable_time:
    facts_maxlen += 1

story_maxlen = max(map(len, (x for x, _, _ in train_facts + test_facts)))
query_maxlen = max(map(len, (x for _, x, _ in train_facts + test_facts)))

vocab_answer = sorted(reduce(lambda x, y: x | y, (set([answer]) for _, _, answer in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')

story_maxlen = MXLEN
if enable_time:
    vocab_size += story_maxlen
    
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
word_idx_answer = dict((c, i) for i, c in enumerate(vocab_answer))

print('Build model...')

#------ Model Definition ---------

def sentence_encoder(input_layer, use_lstm=False, seq_lstm=None):
    '''
        Encodes input
        if use_lstm: Process the sentence with a RNN
    '''
    layer_encoder = Embedding(input_dim=vocab_size,
                            output_dim=EMBED_HIDDEN_SIZE,
                            input_length=story_maxlen,
                            init='glorot_normal')
    input_encoder = layer_encoder(input_layer)

    input_encoder = Lambda(lambda x: x, 
                             output_shape=(story_maxlen, facts_maxlen, EMBED_HIDDEN_SIZE,))(input_encoder)
    position_encoding = input_encoder
    if not use_lstm:
        position_encoding = Lambda(lambda x: K.sum(x, axis=2), 
                                   output_shape=(story_maxlen, EMBED_HIDDEN_SIZE,))(position_encoding)
    else:
        if seq_lstm:
            position_encoding = TimeDistributed(seq_lstm)(position_encoding)
        else:
            seq_lstm = GRU(LSTM_HIDDEN_UNITS, return_sequences=False, init='glorot_normal')
            position_encoding = TimeDistributed(seq_lstm)(position_encoding)

    return position_encoding, layer_encoder, seq_lstm

def input_module(x, u, adjacent=None, use_lstm=False, seq_lstm=None):
    '''
      Process input and create memories to be stored in short-term storage
    '''
    if adjacent == None:
        layer_encoder_m = Embedding(input_dim=vocab_size,
                                  output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=story_maxlen,
                                  init='glorot_normal')

        if use_lstm:
            if seq_lstm:
                input_encoder_m, layer_encoder_m,_ = sentence_encoder(x, use_lstm=True,
                                                                      seq_lstm=seq_lstm)
            else:
                input_encoder_m, layer_encoder_m, seq_lstm = sentence_encoder(x, 
                                                                             use_lstm=True,
                                                                             seq_lstm=None)
        else:
            input_encoder_m, layer_encoder_m,_ = sentence_encoder(x)
    else:
        input_encoder_m, layer_encoder_m = adjacent

    layers = (input_encoder_m, layer_encoder_m)
    
    return layers, seq_lstm

def attention_module(memories, u, use_softmax=True, MLP=None, n_heads=1):
    '''
    Multi-head attention mechanism implementation
    '''
    
    head_outs = []
    mems = []
    for k in range(n_heads):
        # results are similar for linear or tanh activation
        head = TimeDistributed(Dense(LSTM_HIDDEN_UNITS,  W_regularizer=l2(1e-3),
                                     activation='tanh', bias=False))(memories)
        memory = merge([head, u],
                       mode='dot',
                       dot_axes=[2, 1])


        layer_memory = Lambda(lambda x: K.softmax(x/(np.sqrt(LSTM_HIDDEN_UNITS, dtype='float32'))))

        memory = layer_memory(memory)

        head_output = merge([memory, memories],
                      mode = 'dot',
                      dot_axes=[1,1])  
        mems.append(memory)
        head_outs.append(head_output)

    memories_2 = merge(head_outs, mode='concat', concat_axis=-1)

    output= Dense(LSTM_HIDDEN_UNITS, W_regularizer=l2(1e-3), 
                  bias=False, init='glorot_normal')(memories_2)
    output_mem = output
    if MLP:
        output_mem = MLP(output_mem)

    return output, output_mem, mems
    
def get_MLP_t(n):
    '''
    Transition network definition
    '''
    r = []
    for k in range(n):
        size = LSTM_HIDDEN_UNITS if k != 0 or n == 1 else LSTM_HIDDEN_UNITS // 2
        s = stack_layer([
            Dense(size, W_regularizer=l2(1e-3), bias=True, init='glorot_normal'),
            Activation('linear')
        ])
        r.append(s)
    return stack_layer(r)

def get_MLP_g(n):
    '''
    g network for the reasoning module
    '''
    r = []
    for k in range(n):
        size = G_HIDDEN_UNITS
        s = stack_layer([
            Dense(size, W_regularizer=l2(1e-3), bias=True, init='glorot_normal'),
            Activation('relu')
        ])
        r.append(s)
    return stack_layer(r)

def reasoning_module(working_buffer):
    '''
    Implementation of the relational reasoning module
    '''
    
    relations = []
    for fact_object_1 in working_buffer:
        for fact_object_2 in working_buffer:
            relations.append(merge([fact_object_1, fact_object_2, question_encoder], mode='concat',
                                  output_shape=(None, LSTM_HIDDEN_UNITS * 3,)))
    g_MLP_j = get_MLP_g(3) # g network
    mid_relations = []

    for r in relations:
        mid_relations.append(g_MLP_j(r))
    combined_relation = merge(mid_relations, mode='sum')
    return combined_relation

#---------- Building the model ------------------


fact_input = Input(shape=(story_maxlen, facts_maxlen, ), dtype='int32', name='facts_input')
question_input = Input(shape=(query_maxlen, ), dtype='int32', name='query_input')

# Question encoding
question_encoder = Embedding(input_dim=vocab_size,
                               output_dim=EMBED_HIDDEN_SIZE,
                               input_length=query_maxlen,
                               init='glorot_normal')(question_input)

question_encoder = GRU(LSTM_HIDDEN_UNITS, return_sequences=False, 
                       init='glorot_normal')(question_encoder)

# Input encoding

layers, seq_lstm = input_module(fact_input, question_encoder,
                                use_lstm=True, seq_lstm=None)
memories, layers_memories = layers

# Multiple Hops

MLP = get_MLP_t(2) # Transition network

o0_ = MLP(question_encoder)

n_heads = 8

o1,o1_, mem1 = attention_module(memories, o0_, use_softmax=True,
                       MLP=MLP, n_heads=8)

memories1 = memories

o2,o2_, mem2 = attention_module(memories1, o1_, use_softmax=True,
                       MLP=MLP, n_heads=8)

memories2 = memories

o3,o3_, mem3 = attention_module(memories2, o2_, use_softmax=True,
                       MLP=MLP, n_heads=8)

memories3 = memories

o4,o4_, mem4 = attention_module(memories3, o3_, use_softmax=True,
                       MLP=MLP, n_heads=8)

# Reasoning module
working_buffer = [o1,o2,o3,o4]
output = reasoning_module(working_buffer)

response = Dense(len(vocab_answer), activation='softmax', bias=False,
                W_regularizer=l2(1e-3), init='glorot_normal')(output)

# --------- Compiling Model -----------------
# sModel is a modification that avoid batch averaging
model = sModel(input=[fact_input, question_input], output=[response])


sgd = SGD(clipnorm=40.)
adam = Adam(lr=LEARNING_RATE, clipnorm=40.)

print('Compiling model...')

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
if RUN_DEC > 0:
    save_old_str = '{0}_{1}_{2}_{3}_{4}'.format(EMBED_HIDDEN_SIZE, LSTM_HIDDEN_UNITS, MXLEN, RUN_DEC-1, seed)
    model.load_weights('models/weights_memn2n_ntm3_multifocus_mheads_{0}.hdf5'.format(save_old_str))

print('Compilation done...')
def scheduler(epoch):
    if (epoch + 1) % 5 == 0:
        lr_val = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(np.array(lr_val*0.5,dtype='float32'))
    return float(model.optimizer.lr.get_value())

print ('Training... ')
train_split = 0.9
if RUN_DEC > 0:
    train_split = 1. # I use all training data for the extra refining

train_facts = np.array(train_facts)
for k, challenge in enumerate(challenge_files):
    train_values_tmp = train_facts[k*10000:(k+1)*10000]
    choices = np.random.choice(len(train_values_tmp), len(train_values_tmp), replace=False)
    if k == 0:
        valid_splits_sets = train_values_tmp[choices[-int(len(train_values_tmp)*0.1):]]
        train_splits_sets = train_values_tmp[choices[:int(len(train_values_tmp)*train_split)]]
    else:
        valid_splits_sets = np.vstack((valid_splits_sets, 
                                      train_values_tmp[choices[-int(len(train_values_tmp)*0.1):]]))
        train_splits_sets = np.vstack((train_splits_sets,
                                       train_values_tmp[choices[:int(len(train_values_tmp)*train_split)]]))    

train_facts = train_splits_sets[np.random.choice(len(train_splits_sets), 
                                                 len(train_splits_sets), replace=False)]
valid_facts = valid_splits_sets[np.random.choice(len(valid_splits_sets),
                                                 len(valid_splits_sets), replace=False)]

print('Training Size, Valid Size')
print(len(train_facts), len(valid_facts))

inputs_valid, queries_valid, answers_valid = vectorize_facts(valid_facts, word_idx, 
                                                             story_maxlen, query_maxlen, facts_maxlen,
                                                             word_idx_answer=word_idx_answer, enable_time=enable_time)

BATCH_SIZE = 32

show_batch_interval = 1000
N_BATCHS = len(train_facts) // BATCH_SIZE

save_hist = []
save_hist.append(0.)


# ------------ Training Loop ---------------

for k in xrange(EPOCHS):
    choices = np.random.choice(len(train_facts), len(train_facts), replace=False)
    train_facts = train_facts[choices]
    for b,batch in enumerate(grouper(train_facts, BATCH_SIZE, fillvalue=train_facts[-1])):
        start = time.time()
        inputs_train, queries_train, answers_train = vectorize_facts(batch, word_idx, 
                                                                     story_maxlen, query_maxlen, facts_maxlen,
                                                                     word_idx_answer=word_idx_answer, enable_time=enable_time)

        loss = model.train_on_batch([inputs_train, queries_train], 
                                    answers_train)
        end = time.time()
        if b % show_batch_interval == 0:
            print('Epoch: {0}, Batch: {1}, loss: {2} - acc: {3}, time: {4:.3f}s'.format(k, 
                                                                        b, float(loss[0]), float(loss[1]),
                                                                        (end-start)*1000.))

    losses = model.evaluate([inputs_valid, queries_valid], 
                            answers_valid, batch_size=BATCH_SIZE, 
                            verbose=0)
    
    print('Epoch {0}, valid loss / valid accuracy: {1} / {2}'.
           format(k, losses[0], losses[1]))
  
    # Saving model
    if max(save_hist) < losses[1]:
        model.save_weights('models/weights_memn2n_ntm3_multifocus_mheads_{0}.hdf5'.
                           format(save_str), overwrite=True)
    save_hist.append(losses[1])
    if RUN_DEC > 0:
        scheduler(k)

# -------------- Test evaluation ----------------

print('Evaluating model...')

model.load_weights('models/weights_memn2n_ntm3_multifocus_mheads_{0}.hdf5'.format(save_str))

inputs_test, queries_test, answers_test = vectorize_facts(test_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                         word_idx_answer=word_idx_answer, enable_time=enable_time)

print('Total Model Accuracy: ')
loss, acc = model.evaluate([inputs_test, queries_test], 
                           answers_test, verbose=2)
print('Loss: {0}, Error: {1:.1f}'.format(loss, 100.*(1.-acc)))
print('Per-Task Accuracy: ')

passed = 0
total_acc = 0.
for k, challenge in enumerate(challenge_files):
    print(challenge)   
    inputs_test_p_s = inputs_test[k*1000:(k+1)*1000]
    queries_test_p_s = queries_test[k*1000:(k+1)*1000]
    answers_test_s = answers_test[k*1000:(k+1)*1000]

    loss, acc = model.evaluate([inputs_test_p_s, queries_test_p_s], 
                               answers_test_s,  verbose=2)
    total_acc += acc
    print('Loss: {0}, Error: {1:.1f}, Pass: {2} \n'.format(loss, (1.-acc)*100., acc >= 0.95))
    passed += acc >= 0.95
print ('Passed: {0}'.format(passed))
print ('Total acc: {0}'.format(total_acc / len(challenge_files)))
