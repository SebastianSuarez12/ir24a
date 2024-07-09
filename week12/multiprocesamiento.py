import multiprocessing
from multiprocessing import Pool
from random import randrange
import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

model_path = 'week10/data/GoogleNews-vectors-negative300.bin.gz'

# Cargar el modelo Word2Vec preentrenado
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def f(i):
    time.sleep(randrange(0,1))
    print(i*i)

def mq(x):
    for i in range(1,len(x)):
        if x[i] < x[i-1]:
            return False
        
def generate_word2vec_embeddings(text):

    embeddings = []
    tokens = text.lower().split()
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if word_vectors:
        embeddings.append(np.mean(word_vectors, axis=0))
    else:
        embeddings.append(np.zeros(word2vec_model.vector_size))
    return np.array(embeddings)

def print10(x):
    print(x[:100])



if __name__ == '__main__':
    x = range(100)

    df = pd.read_csv('week11/data/podcastdata_dataset.csv')
    corpus = df["text"]

    pool = Pool(processes=8)
    #ans = pool.map(f, x)
    #mq(ans)
    #print(ans)

    #print(pool.map(print10,corpus))

   
    embeddings = pool.map(generate_word2vec_embeddings,corpus[:8])
    print(embeddings)
    #df['embeddings'] = embeddings

    #df.to_csv('week12/data/podcastdata_dataset_emb.csv', index=False)

