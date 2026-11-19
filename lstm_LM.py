import time
import math
import random
import sys
from collections import defaultdict
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import lstm_LM_generate

start = time.time()


class LM_LSTM(nn.Module):
    def __init__(self, nwords, embed_size, hidden_size):  # nwords = vocab size
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(nwords,embed_size)  # embedding layer to convert word indices into embeddings of size embed_size
        self.embedding_dropout = nn.Dropout(p=0.5) # dropout regularizer
        self.rnn = nn.LSTM(embed_size,hidden_size)  # takes word embeddings as input and outputs hidden states of size hidden_size
        self.fc = nn.Linear(hidden_size,nwords)  # logits (unnormalized probabilities) over the entire vocabulary for predicting the next word

    def forward(self, x, h=None):
        x = self.embedding(x)  # convert word indices in x into embeddings and store results in x
        x = self.embedding_dropout(x) # dropout
        x, h = self.rnn(x, h)  # x is the sequence of outputs, h a tuple (h_n, c_n) representing the final hidden and cell state
        x = self.fc(x)  # Pass the LSTM's outputs through the fully connected layer to get the prediction for the next words.
        return x, h


def read(fname, split):
    with open(fname, "r") as fh:
        for line in fh:
            if split == "train":
                sent = [w2i[x] for x in line.strip().split()]
            else:
                sent = [w2i[x] for x in line.strip().split() if x in w2i]
                if len(sent)==0:
                    continue
            sent.append(w2i[EOS])
            yield sent  # more memory-efficient and works well with huge files

# TODO: modify to batchify the data
# Efficient utilization of GPUs
# Stable gradients
# Introduces noise in the gradient updates, which can have a slight regularization effect or prevent overfitting
def calc_lm_loss(batch):
    # model.zero_grad()
    # sent_tensor = torch.tensor(sent, dtype=torch.long).to(device)
    # inputs = sent_tensor[:-1]
    # targets = sent_tensor[1:]  # [len(sent)-1]

    max_len = max(len(sent) for sent in batch)
    batch_size = len(batch)

    inputs = torch.full((max_len -1, batch_size), S, dtype = torch.long, device = device)
    targets = torch.full((max_len -1, batch_size), -100, dtype = torch.long, device=device)

    for i, sent in enumerate(batch):
        length = len(sent)-1
        if length == 0:
            continue
        inputs[:length, i] = torch.tensor(sent[:-1], dtype=torch.long, device=device)
        targets[:length, i]= torch.tensor(sent[1:], dtype = torch.long, device=device)

    outputs, _ = model(inputs)  # [len(sent)-1, nwords]
    loss = criterion(outputs.reshape(-1, nwords), targets.reshape(-1))
    return loss, (targets != -100).sum().item()


# format of files: each line is "word1 word2 ..."
train_file = "ptb/train.txt"
dev_file = "ptb/valid.txt"
# train_file = "twitter/train.txt"
# dev_file = "twitter/dev.txt"


w2i = defaultdict(lambda: len(w2i))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EOS = "<eos>"
train = list(read(train_file, 'train'))
nwords = len(w2i)  # vocab size
dev = list(read(dev_file, 'dev'))
S = w2i[EOS]
assert (nwords == len(w2i))  # make sure no new words were added after reading dev data
print("vocab size %d" % nwords)

EMBED_SIZE = 100
HIDDEN_SIZE = 64
clip_value = 5.0  # for gradient clipping; can be set empirically --> better idea: treat it as a hyperparameter and tune it.
BATCH_SIZE = 32

model = LM_LSTM(nwords, EMBED_SIZE, HIDDEN_SIZE).to(device)
criterion = nn.CrossEntropyLoss(reduction="sum") # use of "sum" is crucial for accurate computation of perplexity. default is "avg." 
optimizer = optim.Adam(model.parameters())

percent_epoch = max(1, math.ceil(len(train) / (2 * BATCH_SIZE))) # to print out results every half epoch. 
print("startup time: %r" % (time.time() - start))
for epoch in range(50):
    model.train()
    print(f"epoch {epoch}:")
    i = this_tokens = this_loss = 0
    random.shuffle(train)
    for start_idx in range(0, len(train), BATCH_SIZE):
        batch = train[start_idx:start_idx + BATCH_SIZE]
        i += 1
        if i % percent_epoch== 0:
            model.eval()
            dev_loss = dev_tokens = 0
            with torch.no_grad(): # the code inside the block won't store the graph history for backpropagation
                for start_idx in range(0, len(dev), BATCH_SIZE):
                    batch = dev[start_idx:start_idx + BATCH_SIZE]
                    loss, mb_tokens = calc_lm_loss(batch)
                    dev_loss += loss.item()
                    dev_tokens += mb_tokens

            model.train()
            print(f" train_loss={this_loss / this_tokens:.4f}, dev_loss={dev_loss / dev_tokens:.4f}, dev_perplexity={math.exp(dev_loss / dev_tokens):.4f}, dev_words={dev_tokens}")  
            # ppl (perplexity): ppl =  exp(-1/|V| sum_{i=1...|V|}(log P(w_i | w_1...w_{i-1}))); 
            # intuitively, ppl of 40-60 (very good ppl on PTB) means that, on average, for each word the model predicts, its uncertainty is roughly equivalent to choosing uniformly from 40-60 wordsin |V|.
            this_loss = this_tokens= 0

        optimizer.zero_grad()
        loss, mb_tokens = calc_lm_loss(batch)
        this_loss += loss.item()
        this_tokens += mb_tokens
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_value) # gradient clipping
        optimizer.step()

# torch.save(model, 'model_LM_PTB.pt')
# model = torch.load('model_LM_PTB.pt')
lstm_LM_generate.generate_sentence_greedy(model, w2i, init_word='cat', count_sentences=1)
lstm_LM_generate.generate_sentence_sampling(model, w2i, init_word='cat')

# ideas for improvement:
# batching
# lr scheduler
# lstm regularization (dropout)
# hyperparameter optimization
# early stopping & model checkpointing
