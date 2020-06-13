TRAIN_SPLIT=0.5

# training hyperparameters
from evaluate import args

# use a batch size of 32 for training
BATCH_SIZE = args.batch_size

# reserve 5% of the train data for validation
VAL_SPLIT = args.val_split

# minimum frequency required to include a token in the training vocabulary
# every word in the training vocabulary is assigned a unique vector in the embedding layer
# all others are assigned to the <unk> token and get a random vector
# vectors not included in the word embeddings are assigned a random vector
MIN_FREQ = args.min_freq

# number of epochs to train the classifier
NUM_EPOCHS = args.num_epochs

# initial learning rate
INIT_LR = args.init_lr

# learning rate decay step size
STEP = args.step

# learning rate decay
DECAY = args.decay

import torch
from torch import nn
from torchtext import data, vocab
from torch.utils.data import DataLoader
from sklearn import metrics

from collections import Counter

import random
import time
import numpy as np

import os

from utils import preprocess, LABELS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_split(x):
    global TRAIN_SPLIT
    TRAIN_SPLIT = x

def setup(seed, path):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(tokenize=preprocess, batch_first=True, include_lengths=True)
    LABEL = MultiLabelField(dtype = torch.float, batch_first=True, preprocessing = lambda x: x.split(','))
    OG = data.Field(tokenize=lambda x: x, batch_first=True)

    train_data = data.TabularDataset(
        path='data/articles.csv',
        format='csv',
        fields=[('label', LABEL), ('text', TEXT), ('original', OG)],
        skip_header=True)

    train_data, test_data = train_data.split(split_ratio=TRAIN_SPLIT, random_state=random.seed(seed))
    
    print(f'finished setup for seed {seed} with data {path}')
    
    return TEXT, LABEL, train_data, test_data
    
class MultiLabelField(data.LabelField):
    
    def build_vocab(self, labels, dataset):
        # do this so that all labels are present even if they incidentally aren't included in the data
        counter = Counter({label: 0 for label in labels})
        for example in dataset.examples:
            counter.update(example.label)
        self.vocab = vocab.Vocab(counter, specials=labels)
        
        weights = []
        factor = 0
        for label in self.vocab.itos:
            freq = self.vocab.freqs[label]
            weight = 1 / self.vocab.freqs[label] if freq else 0
            factor += weight
            weights.append(weight)
        self.weights = torch.tensor(weights) / factor
        
    def numericalize(self, arr, device=None):
        """
        Convert a batch arr into a tensor.
        """
        
        var = torch.zeros([len(arr), len(self.vocab)], dtype=self.dtype, device=device)
        
        for i, example in enumerate(arr):
            for label in example:
                var[i][self.vocab.stoi[label]] = 1.0
        
        return var
    
def run(seed, split, analysis, DIR, folder, dim, TEXT, LABEL, train_data, test_data, randomize=False, saved=False):

    torch.manual_seed(seed)
    
    train_data, val_data = train_data.split(split_ratio=1-VAL_SPLIT, random_state=random.seed(seed))
    
    def gen_batch(batch):
        """
        Formats batch data with offsets.
        """
        text, offsets, labels = [], [0], []

        for example in batch:
            text.extend([TEXT.vocab.stoi[word] for word in example.text])
            offsets.append(len(example.text))
            labels.append(example.label)

        text = torch.tensor(text)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        labels = LABEL.numericalize(labels)

        return text, offsets, labels

    def metric(pred, y, threshold=0):
        y = y.bool().cpu()
        pred = (pred > 0).bool().cpu()
        
        acc = metrics.accuracy_score(y, pred)
        ham = metrics.hamming_loss(y, pred)
        return acc, ham

    def train(data):
        """
        Train one epoch.
        """
        epoch_loss = 0
        epoch_acc, epoch_ham = 0, 0

        #set the model in training phase
        model.train()

        data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=gen_batch)

        for text, offsets, labels in data:
            #resets the gradients after every batch
            optimizer.zero_grad()

            #put data on correct device
            text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)

            output = model(text, offsets)
            loss = criterion(output, labels)
            acc, ham = metric(output, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_ham += ham

        scheduler.step()

        size = len(data)  
        return epoch_loss / size, epoch_acc / size, epoch_ham / size

    def test(data):
        """
        Test one epoch.
        """
        epoch_loss = 0
        epoch_acc, epoch_ham = 0, 0

        #Set the model in evaluation phase
        model.eval()

        data = DataLoader(data, batch_size=len(data), collate_fn=gen_batch)

        for text, offsets, labels in data:
            text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
            with torch.no_grad():
                output = model(text, offsets)

                epoch_loss += criterion(output, labels).item()
                acc, ham = metric(output, labels)
                epoch_acc += acc
                epoch_ham += ham

        size = len(data)
        return epoch_loss / size, epoch_acc / size, epoch_ham / size

    class FeedForward(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes, copy_from=None):
            """
            If copy_from is provided, then vectors for unknown words are copied from
                corresponding rows in copy_from. Otherwise, they are set to 0 vectors.
            """
            super().__init__()
            
            #Pass in a tensor containing the indices to extract from the lookup table
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
            # Load in pretrained embeddings
            if copy_from is not None:
                self.embedding.weight.data.copy_(copy_from)
                if folder is not None:
                    for i, row in enumerate(TEXT.vocab.vectors):
                        if (row != 0).any():
                            self.embedding.weight.data[i].copy_(row)
            elif folder is not None:
                self.embedding.weight.data.copy_(TEXT.vocab.vectors)
            # Lock embedding layer
            self.embedding.weight.requires_grad = False
            
            self.fc = nn.Linear(embed_dim, num_classes)
            
        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            return self.fc(embedded)
    
    if folder is not None:
        name = 'glove.6B' if folder == 'glove.6B' else 'custom'
        VECTORS = vocab.Vectors(name=f'{name}.{dim}d.txt', cache=folder)
    else:
        VECTORS = None
    TEXT.build_vocab(train_data, min_freq=MIN_FREQ, vectors=VECTORS)
    LABEL.build_vocab(LABELS, train_data)

    VOCAB_SIZE = len(TEXT.vocab)
    NUM_CLASSES = len(LABEL.vocab)
    EMBED_DIM = dim
    
    if randomize and folder is not None:
        #generate random weights based on gaussian with mean = VECTORS mean and std = VECTORS std
        mean = VECTORS.vectors.mean(dim=0)
        std = ((VECTORS.vectors - mean).pow(2).sum(dim=0) / VECTORS.vectors.shape[0]).pow(0.5)
        rand_weights = torch.empty(TEXT.vocab.vectors.shape).T
        for i in range(rand_weights.shape[0]):
            rand_weights[i].normal_(mean=mean[i], std=std[i])
        rand_weights = rand_weights.T
    else:
        rand_weights = None
    
    NAME = f'{folder}.{dim}d_{seed}{split}'
    model = FeedForward(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, copy_from=rand_weights).to(device)
    if saved:
        model.load_state_dict(torch.load(f'models/{NAME}'))

    # Use BCEWith logits since it applies sigmoid first.
    criterion = nn.BCEWithLogitsLoss(weight=LABEL.weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEP, gamma=DECAY)
    
    if not saved:
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc, train_ham = train(train_data)
            valid_loss, valid_acc, valid_ham = test(val_data)
        
    test_loss, test_acc, test_ham = test(test_data)
    sigmoid = nn.Sigmoid()
    for ex in test_data:
        analysis['embedding'].append(NAME)
        
        label = list(ex.label)
        label.sort()
        analysis['truth'].append(','.join(label))
        
        tokens = torch.tensor([TEXT.vocab.stoi[token] for token in ex.text]).to(device)
        offset = torch.tensor([0]).to(device)
        probabilities = model(tokens, offset).squeeze()
        analysis[LABELS].append(sigmoid(probabilities).tolist())
        preds = torch.where(probabilities > 0)[0]
        preds = [LABEL.vocab.itos[x] for x in preds]
        preds.sort()
        analysis['pred'].append(','.join(preds))
        
        analysis['text'].append(ex.original)
        
    if not saved:
        if not os.path.isdir('models'):
            os.mkdir('models')
        torch.save(model.state_dict(), f'models/{NAME}')
        
    return np.array([test_loss, test_acc, test_ham])
