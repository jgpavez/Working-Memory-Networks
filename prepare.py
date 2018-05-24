import json
import numpy as np
import os
from PIL import Image
from keras.layers import Embedding
import keras.preprocessing.text

from autocorrect import spell
import re

'''
    Some utility methods to process NLVR.
'''

def correction(word):
    if not str(word).isdigit():
        return spell(word)
    else:
        return word

# This is due to problems with keras text_to_word_sequence
def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]
    
from keras.preprocessing.text import Tokenizer

keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

tokenizer = Tokenizer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def process_sentence(text, objects='False'):
    '''
    Simple and dirty text preprocessing to remove some mispelled words 
    and lemmatize
    '''
    text = text.lower()
    old_text = text
    
    text = text.replace('1', 'one').replace('2','two').replace(
        '3','three').replace('4','four').replace('5','five').replace('6','six').replace(
        '.','').replace('contains', 'contain').replace(
        'which','').replace('are there','there are').replace(
        'there is', '').replace('ablue', 'a blue').replace(
        'corner','edge').replace('wall', 'edge').replace('yelow', 'yellow').replace(
        'below','beneath').replace(
        'brick','block').replace('leats','least').replace('is touching', 'touching')
    text = re.sub(r'colour([\W])', 'color ', text)
    text = re.sub(r'colored([\W])', 'color ', text)
    text = re.sub(r'coloured([\W])', 'color ', text)
    text = text.split(' ')
    text = map(correction, [t for t in text if t])
    text = [lemmatizer.lemmatize(x) if not x in [u'as',u'wall'] else x for x in text]
    text = ' '.join(text)
    if 'that' in text:
        text = text.replace('that', '')
    if 'contain' in text or 'ha ' in text:
        text = text.replace('contain', 'with').replace('ha ','with ')
    text = re.sub(r'(^|\W)a([\W])', ' one ', text)
    text = re.sub(r'(^)ll ', ' ', text)
    text = re.sub(r'(^)t ', 'at ', text)
    text = ' '.join([t for t in text.split(' ') if t])
    text = text.replace('based', 'base')
    return text

def load_data(path, objects=False):
    '''
    Loading NLVR dataset
    '''
    f = open(path, 'r')
    data = []
    for l in f:
        jn = json.loads(l)
        s = process_sentence(jn['sentence'])
        idn = jn['identifier']
        la = int(jn['label'] == 'true')
        if objects:
            la = [int(o in s) for o in objects_list]
            if sum(la) == 0:
                continue
        data.append([idn, s, la])
    return data

def init_tokenizer(sdata):
    texts = [t[1] for t in sdata]
    tokenizer.fit_on_texts(texts)

def tokenize_data(sdata, mxlen):
    texts = [t[1] for t in sdata]
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, mxlen)
    data = {}
    for k in range(len(sdata)):
        data[sdata[k][0]] = [seqs[k], sdata[k][2]]
    return data

def section_image(img):
    '''
    Separating the image into 3 sub-images
    '''
    img1 = img[:,:50,:]
    img2 = img[:,75:125,:]
    img3 = img[:,150:200,:]
    return img1, img2, img3

def load_images(path, sdata, section=False, standarize=False):
    '''
    Load images from NLVR
    '''
    data = {}
    cnt = 0
    for lists in os.listdir(path):
        p = os.path.join(path, lists)
        for f in os.listdir(p):
            cnt += 1
            im_path = os.path.join(p, f)
            im = Image.open(im_path)
            im = im.convert('RGB')
            im = im.resize((200, 50))
            im = np.array(im)
            #im = img_to_array(im)
            if standarize:
                im = tf_image.per_image_standardization(im)
            if section:
                im = section_image(im)
            idf = f[f.find('-') + 1:f.rfind('-')]
            if not idf in sdata:
                continue
            data[f] = [im] + sdata[idf]
    ims, ws, labels = [], [], []
    for key in data:
        ims.append(data[key][0])
        ws.append(data[key][1])
        labels.append(data[key][2])
    data.clear()
    if section:
        ims = np.array(ims, dtype=np.float32)
    ws = np.array(ws, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return ims, ws, labels
