import pandas as pd
import os
from collections import OrderedDict, defaultdict
import random
import numpy as np
import pickle
from classifier import setup, run, set_split
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--val_split', default=0.05, type=float)
parser.add_argument('--min_freq', default=1, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--init_lr', default=0.05, type=float)
parser.add_argument('--step', default=2, type=int)
parser.add_argument('--decay', default=0.9, type=float)
parser.add_argument('--replicate', action='store_true')
args = parser.parse_args()

SAVED = False
RANDOMIZE=True
VECTORS = ['glove.6B', 'custom']
METRICS = ['loss', 'accuracy', 'ham']
DIMS = [50, 100, 200, 300]
DIR = 'results'

cols = OrderedDict()
cols['embedding'] = []
for metric in METRICS:
    cols[metric] = []

if not os.path.exists(os.path.join('results', DIR)):
    os.mkdir(os.path.join('results', DIR))
    
def get_seeds(n, bound=int(1e10)):
    return random.sample(range(0, bound), n)

#SEEDS = get_seeds(5)
if args.replicate:
    SEEDS = [6742404301, 491643334, 2583619689, 9051856379, 6961591582]
else:
    SEEDS = get_seeds(5)

#data_path = 'articles.csv' if args.full else 'split80/test.csv'
data = {seed: setup(seed, path=f'data/articles.csv') for seed in SEEDS}

from utils import LABELS
analysis = {'embedding': [], 'truth': [], 'text': [], 'pred': [], LABELS: []}
    
for dim in DIMS:
    for vector in VECTORS:
        #cols['embedding'].append(f'{dim}d_{vector}')
        results = np.zeros(len(METRICS), dtype='float')
        for seed in SEEDS:
        
            TEXT, LABEL, train_data, test_data = data[seed]
        
            if vector == 'RANDOM':
                test_results_a = run(seed, 'a', analysis, DIR, None, dim, TEXT, LABEL, train_data, test_data, randomize=True, saved=SAVED)
                test_results_b = run(seed, 'b', analysis, DIR, None, dim, TEXT, LABEL, test_data, train_data, randomize=True, saved=SAVED)
            else:   
                test_results_a = run(seed, 'a', analysis, DIR, vector, dim, TEXT, LABEL, train_data, test_data, randomize=RANDOMIZE, saved=SAVED)
                test_results_b = run(seed, 'b', analysis, DIR, vector, dim, TEXT, LABEL, test_data, train_data, randomize=RANDOMIZE, saved=SAVED)
                    
            cols['embedding'].append(f'{vector}.{dim}d_{seed}a')
            cols['embedding'].append(f'{vector}.{dim}d_{seed}b')
            for j, metric in enumerate(METRICS):
                cols[metric].append(test_results_a[j])
                cols[metric].append(test_results_b[j])
        print(f'finished {vector}.{dim}d')
        
df = pd.DataFrame(cols)
with open(os.path.join('results', DIR, 'results.pkl'), 'wb') as f:
    pickle.dump(df, f)
with open(os.path.join('results', DIR, 'results.txt'), 'w') as f:
    f.write(df.to_string(index=False))

analysis = pd.DataFrame(analysis)
with open(os.path.join('results', DIR, 'analysis.pkl'), 'wb') as f:
    pickle.dump(analysis, f)
