from utils import preprocess
from glove import Corpus, Glove
import pandas as pd
import time
import random
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=25, type=int)
args = parser.parse_args()

DATA = 'data/articles.csv'
PATH = f'custom'
CORPUS_PATH = os.path.join(PATH, 'corpus.pkl')
SEED = 34985734958
epochs = args.epochs

if not os.path.isdir(PATH):
    os.mkdir(PATH)

if os.path.exists(CORPUS_PATH):
    print('Found existing corpus.')
    corpus = Corpus.load(CORPUS_PATH)
else:
    print('Could not find existing corpus. Creating new one.')
    class Iterable:
        def __init__(self, df, col='text'):
            self.df = df
            self.col = col
            
        def __iter__(self):
            for article in self.df[self.col].values:
                yield preprocess(article)

    corpus = Corpus()
    start = time.time()
    corpus.fit(Iterable(pd.read_csv(DATA)))
    print(f'finished co_occur in {int(time.time() - start)} seconds.')
    corpus.save(CORPUS_PATH)

def train_dim(size):
    """
    Trains and saves a SIZE-dimensional glove embedding.
    """
    glove = Glove(no_components=size, random_state=random.seed(SEED))
    start = time.time()
    glove.fit(corpus.matrix, epochs=epochs, no_threads=12, verbose=True)
    print(f'finished {size}d vectors in {(time.time() - start)/60:.2f} minutes.')
    
    with open(f'{PATH}/custom.{size}d.txt', 'w') as f:
        for word, i in corpus.dictionary.items():
            word += ' '
            # Give each vector value 6 significant figures
            word += ' '.join(f'{x:.6g}' for x in glove.word_vectors[i])
            word += '\n'

            f.write(word)

for size in [50, 100, 200, 300]:
    train_dim(size)
