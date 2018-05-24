import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences

'''
    Some utility methods to process bAbI.
    Some of these methods were based on
    https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py
'''
def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip().lower()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q[:-1], a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append([sent[:-1]])
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_facts(data, word_idx, story_maxlen, query_maxlen, fact_maxlen, word_idx_answer = None,
                    enable_time = False):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = np.zeros((len(story), fact_maxlen),dtype='int32')
        for k,facts in enumerate(story):
            if not enable_time:
                x[k][-len(facts):] = np.array([word_idx[w] for w in facts])[:fact_maxlen]
            else:
                x[k][-len(facts)-1:-1] = np.array([word_idx[w] for w in facts])[:fact_maxlen-1]
                x[k][-1] = len(word_idx) + len(story) - k
        xq = [word_idx[w] for w in query]
        if word_idx_answer:
            y = np.zeros(len(word_idx_answer))
        else:
            y = np.zeros(len(word_idx) + 1) if not enable_time else np.zeros(len(word_idx) + 1 + story_maxlen)
        if word_idx_answer:
            y[word_idx_answer[answer]] = 1
        else:
            y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

