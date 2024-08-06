

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from nltk.stem import PorterStemmer
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer
import gensim
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
import tensorflow as tf
# add to the top of your code under import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

ps = PorterStemmer()
snow_stemmer = SnowballStemmer(language='english')
stemmer = LancasterStemmer()

import spacy
spacy.prefer_gpu()

from sklearn.metrics import f1_score

np.set_printoptions(suppress=True)

%%time

train=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv").fillna(" ")
test=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv").fillna(" ")
train_question=train['question_text']
test_question=test['question_text']

question_list=pd.concat([train_question,test_question])
y=train['target'].values

num_train_data = y.shape[0]


import zipfile

z= zipfile.ZipFile('/kaggle/input/quora-insincere-questions-classification/embeddings.zip')

for x in z.namelist():
    if x.endswith('300d.txt'):
        z.extract(x,"embedded_file")
    elif x.endswith('300d-1M.vec'):
        z.extract(x,"embedded_file")
        


nlp=spacy.load('en_core_web_lg',disable=['tok2vec', 'tagger', 'parser', 'senter', 'attribute_ruler', 'ner'])
docs=nlp.pipe(question_list)
#docs = nlp.pipe(question_list,n_process=-1)

vocab = {}
sentencetoken=[]
index=1
lemmatization={}
for sentence in tqdm(docs):
    sentoken=[]
    for word in sentence:
        if (not word.is_punct) and word.text not in vocab:
        #if (word.pos_  !=  "PUNCT") and (word.text not in 9ocab):
           # print(word.text)
            vocab[word.text] =index
            index+=1
            lemmatization[word.text]=word.lemma_
        if not word.is_punct:
        #if word.pos_ != "PUNCT":
            sentoken.append(vocab[word.text])
    sentencetoken.append(sentoken)
    


del docs

train_seq = sentencetoken[:num_train_data]
test_seq = sentencetoken[num_train_data:]


train_seq=pad_sequences(train_seq,maxlen=55,padding='post')
test_seq=pad_sequences(test_seq,maxlen=55,padding='post')

%%time

spell_model = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/working/embedded_file/wiki-news-300d-1M/wiki-news-300d-1M.vec')
words = spell_model.index_to_key

w_rank = {}

for i,word in enumerate(words):
    w_rank[word] = i
WORDS = w_rank

%%time

# Use fast text as vocabulary
def words(text): return re.findall(r'\w+', text.lower())
def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)
def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)
def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or [word])
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def singlify(word):
    return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])

%%time

def get_embeddings(vocab,file,lemmatization):
    def get_coefs(word,*arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    out_of_vocab={}
    vocab_len=len(vocab)+1
    embed_size=300
    embedding_matrix=np.zeros((vocab_len,embed_size),dtype=np.float32)
    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
    i=1
    for unique in tqdm(vocab):
        word=unique
        embedded_word_vector=embeddings_index.get(word)
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_word_vector
            continue
        word=unique.lower()
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_word_vector
            continue
        word=unique.upper()
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_word_vector
            continue
        word=unique.capitalize()
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_word_vector
            continue
        word = lemmatization[unique]
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_word_vector
            continue_word
        word = ps.stem(unique)
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_word_vector
            continue
        word = snow_stemmer.stem(unique)
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_vector
            continue
        word = stemmer.stem(unique)
        if embedded_word_vector is not None:
            embedding_matrix[vocab[unique]]=embedded_word_vector
            continue
        if len(unique) > 1:
            word = correction(unique)
            embedding_word_vector = embeddings_index.get(word)
            if embedding_word_vector is not None:
                embedding_matrix[vocab[unique]] = embedding_word_vector
                continue
        embedding_matrix[vocab[unique]] = unknown_vector     
        if unique not in out_of_vocab:
            out_of_vocab[unique]=1
        else:
            out_of_vocab[unique]=out_of_vocab.get(unique)+1
    return embedding_matrix,out_of_vocab

%%time
def get_f1score(pred,actual):
    for i in np.arange(0.1,1,0.050):
        f=((pred >i)+0)
        print("f1 score at {:03f} ".format(i),f1_score(actual, f))


%%time

file="/kaggle/working/embedded_file/glove.840B.300d/glove.840B.300d.txt"

Glove_Embedded_Matrix,Glove_Out_of_Vocabulary=get_embeddings(vocab,file,lemmatization)


#file="/kaggle/input/tttttt/wiki-news-300d-1M.vec"

#wiki_Embedded_Matrix,wiki_Out_of_Vocabulary=get_embeddings(vocab,file,lemmatization)


from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D,GlobalMaxPooling3D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.models import Model
from keras import backend as K
#from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Input, Embedding, Dense
import tensorflow 
from tensorflow.keras.layers import  Embedding
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import keras

Embedded_Matrix=Glove_Embedded_Matrix

#Embedded_Matrix=wiki_Embedded_Matrix



# hyperparameters
max_length = 55
embedding_size = 300
learning_rate = 0.001
batch_size = 512
num_epoch = 4


vocab_len=len(vocab)+1
layer=Embedding(vocab_len,embedding_size)
layer.build((None,))
layer.set_weights([Embedded_Matrix])
layer.trainable=False


def get_model():
    model=Sequential()
    model.add(layer)
    model.add(Bidirectional(GRU(256,return_sequences=True)))

    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(1,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model

# model=get_model()
# model.fit(X_train,y_train,batch_size=batch_size,verbose=2,epochs=num_epoch-1)
# pred=model.predict(X_test,batch_size=batch_size)
# #get_f1score(pred,y_test)


modelss=[]
preds=[]
kf = KFold(n_splits=5, shuffle=True, random_state=42069)
for train_idx, val_idx in kf.split(train_seq):
    x_train_f = train_seq[train_idx]
    y_train_f = y[train_idx]
    x_val_f = train_seq[val_idx]
    y_val_f = y[val_idx]

    model = get_model()
    model.fit(x_train_f,y_train_f,batch_size=batch_size,verbose=2,epochs=num_epoch-1,validation_data=(x_val_f, y_val_f))
    pred=model.predict(x_val_f,batch_size=batch_size) 
    get_f1score(pred,y_val_f)
    modelss.append(model)
    preds.append(model.predict(test_seq,batch_size=batch_size))


mean=0
for i in range(0,5):
    mean+=preds[i]
    
mean=mean/5

pred=mean


submission = pd.DataFrame.from_dict({'qid': test['qid']})
submission['prediction'] = (pred>0.35).astype(int)
submission.to_csv('submission.csv', index=False)
