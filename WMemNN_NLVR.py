''' 
  Implementation of a Working Memory Network for the NLVR dataset.
  authors: jgpavez
 
'''

from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, concatenate, add, Dropout, RepeatVector, Lambda, Permute, Activation, Reshape, TimeDistributed, GRU, Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam

import keras.backend as K
from keras.regularizers import l2
from keras.layers.core import Layer

from keras.layers.normalization import BatchNormalization

from functools import reduce
import tarfile
import numpy as np
import random

import sys
import tensorflow as tf
seed = 1234
if len(sys.argv) > 5:
    seed = int(sys.argv[5])

np.random.seed(seed)
random.seed(seed)

print(seed)

import time

import copy
import sys

from itertools import izip_longest

import prepare

import json

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def groupers(iterables, n, fillvalues=None):
    result = []
    for k, iterable in enumerate(iterables):
        args = [iter(iterable)] * n
        result.append(izip_longest(*args, fillvalue=fillvalues[k]))
    return result       

class LayerNorm(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = initializations.one(input_shape[1:], name='gamma')
        self.beta = initializations.zero(input_shape[1:], name='beta')
        self.trainable_weights = [self.gamma, self.beta]

        super(LayerNorm, self).build(input_shape)

    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def get_output_shape_for(self, input_shape):
        return input_shape

class MaskedAttention(Layer):
    def __init__(self, **kwargs):
        super(MaskedAttention, self).__init__(**kwargs)
    
    def call(self, X, mask=None):
        input_shape = X.shape
        
        sorted_values = T.argsort(X, axis=1)[:, :-pooling_size]

        # Construct a row of indexes to the length of axis
        indexes = T.arange(X.shape[1]).dimshuffle(
            *(['x' for dim1 in xrange(1)] + [0] + ['x' for dim2 in xrange(X.ndim - 1 - 1)]))
        sorted_values = sorted_values.dimshuffle(0,1,'x')
        mask = T.eq(indexes, sorted_values).sum(axis=1)
        masked_attention = T.set_subtensor(X[mask.nonzero()], 0)      
        y = masked_attention
        return y

def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f

# Model & Training parameters
EMBED_HIDDEN_SIZE = 100
CNN_EMBED_SIZE = 24
LSTM_HIDDEN_UNITS = 128
mxlen = 50
CNN_FEATURES_SIZE = 7*7

if len(sys.argv) > 1:
    EMBED_HIDDEN_SIZE = int(sys.argv[1])
if len(sys.argv) > 2:
    CNN_EMBED_SIZE = int(sys.argv[2])
if len(sys.argv) > 3:
    LSTM_HIDDEN_UNITS = int(sys.argv[3])
if len(sys.argv) > 4:
    mxlen = int(sys.argv[4])

print(EMBED_HIDDEN_SIZE, CNN_EMBED_SIZE, LSTM_HIDDEN_UNITS,
      mxlen, seed)
    
EPOCHS = 60
BATCH_SIZE = 256

train_json = 'data/nlvr/train/train.json'
train_img_folder = 'data/nlvr/train/images'
dev_json = 'data/nlvr/dev/dev.json'
dev_img_folder = 'data/nlvr/dev/images'

print('Processing input data...')

data = prepare.load_data(train_json)
prepare.init_tokenizer(data)
data = prepare.tokenize_data(data, mxlen)
imgs, ws, labels = prepare.load_images(train_img_folder, data, section=True)
data.clear()

dev_data = prepare.load_data(dev_json)
dev_data = prepare.tokenize_data(dev_data, mxlen)
dev_imgs, dev_ws, dev_labels = prepare.load_images(dev_img_folder, dev_data, section=True)
dev_data.clear()

imgs_mean = np.mean(imgs)
imgs_std = np.std(imgs - imgs_mean)
imgs = (imgs - imgs_mean) / (imgs_std + 1e-7)

dev_imgs = (dev_imgs - imgs_mean) / (imgs_std + 1e-7)

imgs_1, imgs_2, imgs_3 = imgs[:,0,:,:], imgs[:,1,:,:], imgs[:,2,:,:]
dev_imgs_1, dev_imgs_2, dev_imgs_3 = dev_imgs[:,0,:,:], dev_imgs[:,1,:,:], dev_imgs[:,2,:,:]

word_index = prepare.tokenizer.word_index
inv_index = {v:k for k, v in word_index.items()}
def translate(k): return [inv_index[w] for w in k if w != 0]

vocab_size = len(word_index) + 1
print('VOCAB : {0}'.format(vocab_size))

choices = np.random.choice(int(len(imgs_1)), int(len(imgs_1)), replace=False)
imgs_1 = imgs_1[choices]
imgs_2 = imgs_2[choices]
imgs_3 = imgs_3[choices]
ws = ws[choices]
labels = labels[choices]

print('Build model...')

# ----- Model Definition -------
# ----- Convolutional Layer ----
def bn_layer(x, conv_unit):
    def f(inputs):
        md = Conv2D(x, (conv_unit, conv_unit), padding='same', kernel_initializer='he_normal')(inputs)
        #md = BatchNormalization()(md)
        #md = LayerNorm()(md)
        return Activation('relu')(md)

    return f

def conv_net_model():
    inputs = Input((50, 50, 3))
    model = bn_layer(CNN_EMBED_SIZE, 3)(inputs)
    model = MaxPooling2D((3, 3), (3, 3))(model)
    model = bn_layer(CNN_EMBED_SIZE-2, 3)(model)
    model = MaxPooling2D((3, 3), (2, 2))(model)
    out_model = Model(inputs=[inputs], outputs=[model])
    return out_model

def slice_1(t):
    return t[:, 0, :, :]

def slice_2(t):
    return t[:, 1:, :, :]

def slice_3(t):
    return t[:, 0, :]

def slice_4(t):
    return t[:, 1:, :]

def concatenation(x, k1, k2):
    D = K.variable(BATCH_SIZE*[[k1,k2]])

    return K.concatenate([x,D])

def input_module(x, cnn_model):
    '''
      Process input and create memories to be stored in short-term storage
    '''
    cnn_features = cnn_model(x)
    shapes = cnn_features._keras_shape
    w, h = shapes[1], shapes[2]
    print('Objects shape: {0} {1}'.format(w, h))
    CNN_FEATURES_SIZE = w*h
    
    slice_layer1 = Lambda(slice_1, output_shape=(h, CNN_EMBED_SIZE-2, ))
    slice_layer2 = Lambda(slice_2, output_shape=(h, CNN_EMBED_SIZE-2, ))
    slice_layer3 = Lambda(slice_3, output_shape=(CNN_EMBED_SIZE-2,))
    slice_layer4 = Lambda(slice_4, output_shape=(CNN_EMBED_SIZE-2,))
    features = []
    for k1 in range(w):
        features1 = slice_layer1(cnn_features)
        cnn_features = slice_layer2(cnn_features)
        for k2 in range(h):
            features2 = slice_layer3(features1)
            features1 = slice_layer4(features1)
            features2 = Lambda(lambda x: concatenation(x, float(k1)/h, float(k2)/h),
                           output_shape=(CNN_EMBED_SIZE,))(features2)
            features.append(features2)

    for k, m in enumerate(features):
        features[k] = Reshape((1, CNN_EMBED_SIZE), input_shape=(CNN_EMBED_SIZE, ))(m)

    feature = concatenate(features, axis=-2)

    return feature


def attention_module(memories, question_encoder, use_softmax=True, MLP=None, memory_model=None):
    '''
        Additive attention mechanism implementation
    '''

    question_encoder_repeat = RepeatVector(CNN_FEATURES_SIZE)(question_encoder)

    memory_question = concatenate([memories, question_encoder_repeat])

    input_layer = Input(shape=(CNN_EMBED_SIZE + LSTM_HIDDEN_UNITS,))
    out = Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(input_layer)
    out = Dense(128, activation='relu', kernel_regularizer=l2(1e-3))(out)
    out = Dense(1, activation='tanh', kernel_regularizer=l2(1e-3))(out)
    model = Model(inputs=input_layer, outputs=out)

    memory = TimeDistributed(model)(memory_question)
    memory = Reshape((CNN_FEATURES_SIZE,),
                        input_shape=(CNN_FEATURES_SIZE, 1))(memory)

    if use_softmax:
        layer_memory = Lambda(lambda x: K.softmax(x))
    else:
        layer_memory = Lambda(lambda x: x)
    memory = layer_memory(memory)

    output = Lambda(lambda x: tf.einsum('aji,aj->ai', x[0], x[1]),
                   output_shape=(CNN_EMBED_SIZE, ))([memories, memory])
    output_mem = output

    if MLP:
        output_mem = MLP(output_mem)
    return output, output_mem, memory

def get_MLP_t(n):
    r = []
    for k in range(n):
        size = LSTM_HIDDEN_UNITS if k != 0 or n == 1 else LSTM_HIDDEN_UNITS // 2
        s = stack_layer([
            Dense(size, use_bias=True, kernel_initializer='glorot_normal', kernel_regularizer=l2(1e-3)),
            #BatchNormalization(mode=2),
            Activation('relu')
        ])
        r.append(s)
    return stack_layer(r)

def get_MLP_g(n):
    '''
    g network for the reasoning module
    '''
    r = []
    units_mlp = [256,128,64]
    for k in range(n):
        MLP_unit = units_mlp[k]
        s = stack_layer([
            Dense(MLP_unit, use_bias=True, kernel_initializer='glorot_normal'),
            #BatchNormalization(mode=2),
            Activation('relu'),
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
            relations.append(concatenate([fact_object_1, fact_object_2, question_encoder]))
    g_MLP_g = get_MLP_g(3) # g network
    mid_relations = []

    for r in relations:
        mid_relations.append(g_MLP_g(r))
    combined_relation = add(mid_relations)
    return combined_relation

#---------- Building the model ------------------

fact_input_1 = Input((50, 50, 3), dtype='float32', name='fact_input_1')
fact_input_2 = Input((50, 50, 3), dtype='float32', name='fact_input_2')
fact_input_3 = Input((50, 50, 3), dtype='float32', name='fact_input_3')

# Question encoding
question_input = Input(shape=(mxlen, ), dtype='int32', name='query_input')

question_encoder = Embedding(input_dim=vocab_size,
                  output_dim=EMBED_HIDDEN_SIZE,
                  input_length=mxlen,
                  embeddings_initializer='glorot_normal',
                  trainable=True)(question_input)

question_encoder = Lambda(lambda x: x, 
                       output_shape=(mxlen, EMBED_HIDDEN_SIZE,))(question_encoder)

question_encoder = GRU(LSTM_HIDDEN_UNITS, return_sequences=False, 
                       kernel_initializer='glorot_normal')(question_encoder)

# Input encoding
conv_model = conv_net_model()

memories_1 = input_module(fact_input_1, conv_model)
memories_2 = input_module(fact_input_2, conv_model)
memories_3 = input_module(fact_input_3, conv_model)

# Multiple Hops
MLP = get_MLP_t(1)

o0_ = question_encoder

memories = memories_1
o1,o1_, mem1 = attention_module(memories, o0_, use_softmax=True,
                       MLP=MLP)

memories1 = memories_2
o2,o2_, mem2 = attention_module(memories1, o0_, use_softmax=True,
                       MLP=MLP)

memories2 = memories_3
o3,o3_, mem3 = attention_module(memories2, o0_, use_softmax=True,
                       MLP=MLP)

memories = memories_1
o4,o4_, mem4 = attention_module(memories, o1_, use_softmax=True,
                       MLP=None)

memories1 = memories_2
o5,o5_, mem5 = attention_module(memories1, o2_, use_softmax=True,
                       MLP=None)

memories2 = memories_3
o6,o6_, mem6 = attention_module(memories2, o3_, use_softmax=True,
                       MLP=None)


working_buffer = [o1, o2, o3, o4, o5, o6]

MLP_unit = 64

# Reasoning module
output = reasoning_module(working_buffer)

def bn_dense(x, MLP_unit):
    y = Dense(MLP_unit)(x)
    y = Activation('relu')(y)
    return y

rn = bn_dense(output, 32)
pred = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', use_bias=True)(rn)

# --------- Compiling Model -----------------
model = Model(inputs=[fact_input_1, fact_input_2, fact_input_3, question_input], outputs=[pred])

#print(model.summary())

adam = Adam(lr=0.001)

print('Compiling model...')

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print('Compilation done...')

print('Start Training')

show_batch_interval = 50
save_hist = []
save_hist.append(0.)

from keras.utils.np_utils import to_categorical

#------------ Training Loop ------------

for k in xrange(EPOCHS):
    for b, batch in enumerate(zip(*groupers([imgs_1, imgs_2, imgs_3, ws, labels], 
                                       BATCH_SIZE, 
                                       fillvalues=[imgs_1[-1], imgs_2[-1], imgs_3[-1],
                                                   ws[-1], labels[-1]]))):
        start = time.time()
        imgs_batch_1, imgs_batch_2, imgs_batch_3, ws_batch, labels_batch = batch
        imgs_batch_1, imgs_batch_2, imgs_batch_3 = np.array(imgs_batch_1), np.array(imgs_batch_2), np.array(imgs_batch_3)
        ws_batch, labels_batch = np.array(ws_batch), np.array(labels_batch)
        
        loss = model.train_on_batch([imgs_batch_1, imgs_batch_2, 
                                     imgs_batch_3, ws_batch], labels_batch)
        end = time.time()
        if b % show_batch_interval == 0:
            print('Epoch: {0}, Batch: {1}, loss: {2} - acc: {3}, time: {4:.3f}s'.format(k, 
                                                                        b, float(loss[0]), float(loss[1]),
                                                                        (end-start)*1000.))

    # I need to do this in order to avoid problems in concatenation function
    rest = len(dev_imgs_1) % BATCH_SIZE
    losses = model.evaluate([dev_imgs_1[:-rest], dev_imgs_2[:-rest],
                             dev_imgs_3[:-rest], dev_ws[:-rest]], 
                             dev_labels[:-rest], batch_size=BATCH_SIZE, 
                            verbose=0)
    print('Epoch {0}, valid loss / valid accuracy: {1} / {2}'.
           format(k, losses[0], losses[1]))
      
    #Saving model
    if max(save_hist) < losses[1]:
        model.save_weights('models/weights_memn2n_multifocus_nlvr_{0}.hdf5'.format(seed), overwrite=True)
    save_hist.append(losses[1])

# -------------- Test evaluation ----------------
print('Evaluating model...')

model.load_weights('models/weights_memn2n_multifocus_nlvr_{0}.hdf5'.format(seed))    

rest = len(dev_imgs_1) % BATCH_SIZE
losses = model.evaluate([dev_imgs_1[:-rest], dev_imgs_2[:-rest],
                         dev_imgs_3[:-rest], dev_ws[:-rest]], 
                        dev_labels[:-rest], batch_size=BATCH_SIZE, 
                        verbose=0)
print('valid loss / valid accuracy: {0} / {1}'.
      format(losses[0], losses[1]))

test_json = 'data/nlvr/test/test.json'
test_img_folder = 'data/nlvr/test/images'

test_data = prepare.load_data(test_json)
test_data = prepare.tokenize_data(test_data, mxlen)
test_imgs, test_ws, test_labels = prepare.load_images(test_img_folder, test_data, 
                                                         transform=False, section=True)
test_data.clear()

test_imgs = (test_imgs - imgs_mean) / (imgs_std + 1e-7)

test_imgs_1, test_imgs_2, test_imgs_3 = test_imgs[:,0,:,:], test_imgs[:,1,:,:], test_imgs[:,2,:,:]

rest = len(test_imgs_1) % BATCH_SIZE

losses = model.evaluate([test_imgs_1[:-rest], test_imgs_2[:-rest],
                         test_imgs_3[:-rest], test_ws[:-rest]], 
                        test_labels[:-rest], batch_size=BATCH_SIZE, 
                        verbose=0)

print('test loss / test accuracy: {0} / {1} \n'.
              format(losses[0], losses[1]))  
