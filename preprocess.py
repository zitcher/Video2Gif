from torch.utils.data import Dataset, DataLoader
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import random
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from torch import nn, optim
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import pickle

def replaceEl(lst, vals, newval):
    newlst = []
    for item in lst:
        if item in vals:
            newlst.append(newval)
        else:
            newlst.append(item)
    return newlst

def save_obj(obj, name):
    with open('./data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('./data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

class GPT2Dataset(Dataset):
    def __init__(self, input_file, tokenizer):
        """
        Expects a file of where each lines is a sentence of: word1, word2, ... wordk [SEP] vidtoken1, vidtoken2, vidtokenk
        """
        self.vid_vocab = dict()
        self.vocab_vid = dict()
        loaded = False
        if os.path.exists("./data/vid_vocab.pkl") and os.path.exists("./data/vocab_vid.pkl"):
            print("loading vocab")
            self.vid_vocab = load_obj("vid_vocab")
            self.vocab_vid = load_obj("vocab_vid")
            loaded = True

        # read the input file line by line and put the lines in a list.
        with open(input_file) as f:
            self.raw = f.read()

        sentences = self.raw.split('\n')
        for line in tqdm(sentences):
            sentence_vids = line.split('[SEP]')
            if (len(sentence_vids) < 2):
                print("LINE", line)
            vids = sentence_vids[1].strip().split(' ')

            if not loaded:
                if '[SEP]' not in self.vid_vocab:
                        self.vid_vocab['[SEP]'] = len(tokenizer) + len(self.vid_vocab)
                        self.vocab_vid[self.vid_vocab['[SEP]']] = '[SEP]'

                for vid in vids:
                    if vid not in self.vid_vocab:
                        self.vid_vocab[vid] = len(tokenizer) + len(self.vid_vocab)
                        self.vocab_vid[self.vid_vocab[vid]] = vid
        if not loaded:
            print("saving vocab")
            save_obj(self.vid_vocab, "vid_vocab")
            save_obj(self.vocab_vid, "vocab_vid")

        self.input_ids = []
        self.mask = [] # Mask to avoid performing attention on padding token indices. 1 for tokens that are not masked, 0 for tokens that are masked.
        self.labels = [] # -100 are ignored (masked)
        for i in trange(len(sentences), desc='preprocessing'):
            # add a correct next sentence example
            line = sentences[i]
            linesplit = line.split('[SEP]')
            linesplit = [
                self.vectorize_line(linesplit[0].strip(), tokenizer) + [self.vid_vocab['[SEP]']],
                self.vectorize_gif(linesplit[1].strip())
            ]
            
            vector = linesplit[0] + linesplit[1]

            # print(line, vector, self.decode(vector, tokenizer))
            self.input_ids.append(torch.tensor(vector))
            self.labels.append(torch.tensor(linesplit[0] + linesplit[1] + [tokenizer.eos_token_id])) # [-100] *  len(linesplit[0]) + linesplit[1] + [tokenizer.eos_token_id]
            self.mask.append(torch.tensor([1] * len(vector)))

        self.input_ids = pad_sequence(self.input_ids, batch_first=True, padding_value=0)
        self.labels = pad_sequence(self.labels, batch_first=True, padding_value=-100)
        self.mask = pad_sequence(self.mask, batch_first=True, padding_value=0)
    
    def vectorize_gif(self, gif):
        encode = []
        for tok in gif.split(' '):
            encode.append(self.vid_vocab[tok])
        return encode

    def vectorize_line(self, line, tokenizer):
        encode = tokenizer.encode(line)
        return encode

    def decode(self, vector, tokenizer):
        sep = vector.index(self.vid_vocab['[SEP]'])
        gifs = ''
        for vid in vector[sep + 1:len(vector) - 1]:
            gifs += ' ' + self.vocab_vid[vid]

        return tokenizer.decode(vector[:sep + 1]) + gifs

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        :return: dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        item = {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "mask": self.mask[idx],
        }
        return item

if __name__ == "__main__":
    gpt2Tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print("loading dataset")
    train_dataset = GPT2Dataset('./test.tsv', gpt2Tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print("loading modle")
    vocab_size = len(gpt2Tokenizer) + len(train_dataset.vid_vocab)
    model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
    model.resize_token_embeddings(vocab_size)
    
    
    print("testing training")
    optimizer = optim.Adam(model.parameters(), 0.001)
    model = model.train()
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    scheduler = get_linear_schedule_with_warmup(optimizer, 2, 8)
    epochs = 8
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"]
            mask = batch["mask"]
            labels = batch["labels"]

            out = model(
                input_ids=input_ids,
                attention_mask=mask,
                labels=labels,
            )
            logits = out.logits
            optimizer.zero_grad()
            loss = loss_fn(logits.permute(0, 2, 1), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(torch.exp(loss).item())
        print("epoch", epoch, "perplexity:", np.mean(losses))