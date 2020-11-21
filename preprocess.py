from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizerFast, BertForPreTraining
import random
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from torch import nn, optim
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence


def replaceEl(lst, vals, newval):
    newlst = []
    for item in lst:
        if item in vals:
            newlst.append(newval)
        else:
            newlst.append(item)
    return newlst

class BERTDataset(Dataset):
    def __init__(self, input_file, tokenizer):
        """
        Expects a file of where each lines is a sentence of: word1, word2, ... wordk [SEP] vidtoken1, vidtoken2, vidtokenk
        """
        self.vid_vocab = dict()
        self.vocab_vid = dict()

        # read the input file line by line and put the lines in a list.
        with open(input_file) as f:
            self.raw = f.read()

        sentences = self.raw.split('\n')
        for line in tqdm(sentences):
            sentence_vids = line.split('[SEP]')
            vids = sentence_vids[1].strip().split(' ')

            for vid in vids:
                if vid not in self.vid_vocab:
                    self.vid_vocab[vid] = len(tokenizer) + len(self.vid_vocab)
                    self.vocab_vid[self.vid_vocab[vid]] = vid

        # get dataset in a random order, impt for next sent prediction construction
        # since we use the next sentence in the list as the example of an incorrect next sentence prediction
        random.shuffle(sentences)

        self.input_ids = []
        self.lengths = []
        self.max_seq_len = 0
        self.token_type_ids = [] # 0 for sent A, 1 for sent B
        self.labels = [] # -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size]
        self.next_sentence_label = [] # Indices should be in [0, 1], 0 indicates sequence B is a continuation of sequence A
        for i in trange(len(sentences), desc='preprocessing'):
            # add a correct next sentence example
            line = sentences[i]
            linesplit = line.split('[SEP]')
            linesplit = [
                self.vectorize_line(linesplit[0].strip(), tokenizer),
                self.vectorize_gif(linesplit[1].strip()) + [tokenizer.sep_token_id]
            ]
            
            vector = linesplit[0] + linesplit[1]
            self.input_ids.append(torch.tensor(vector))
            self.lengths.append(torch.tensor(len(vector)))
            self.max_seq_len = max(self.max_seq_len, len(vector))
            self.token_type_ids.append(torch.tensor([0] * len(linesplit[0]) + [1] * len(linesplit[1])))

            vector = replaceEl(vector, [tokenizer.cls_token_id, tokenizer.sep_token_id], -100)
            self.labels.append(torch.tensor(vector))
            self.next_sentence_label.append(torch.tensor([0]))

            # add a incorrect next sentence example
            nextline = None
            if i < len(sentences) - 1:
                nextline = sentences[i + 1]
            else:
                nextline = sentences[0]

            nextlinesplit = nextline.split('[SEP]')
            nextlinesplit = [
                linesplit[0],
                self.vectorize_gif(nextlinesplit[1].strip()) + [tokenizer.sep_token_id]
            ]

            vector = nextlinesplit[0] + nextlinesplit[1]
            self.input_ids.append(torch.tensor(vector))
            self.lengths.append(torch.tensor(len(vector)))
            self.max_seq_len = max(self.max_seq_len, len(vector))
            self.token_type_ids.append(torch.tensor([0] * len(nextlinesplit[0]) + [1] * len(nextlinesplit[1])))
            vector = replaceEl(vector, [tokenizer.cls_token_id, tokenizer.sep_token_id], -100)
            self.labels.append(torch.tensor(vector))
            self.next_sentence_label.append(torch.tensor([1]))

        self.input_ids = pad_sequence(self.input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        self.token_type_ids = pad_sequence(self.token_type_ids, batch_first=True, padding_value=1)
        self.labels = pad_sequence(self.labels, batch_first=True, padding_value=-100)
    
    def vectorize_gif(self, gif):
        encode = []
        for tok in gif.split(' '):
            encode.append(self.vid_vocab[tok])
        return encode

    def vectorize_line(self, line, tokenizer):
        encode = tokenizer.encode(line)
        return encode

    def decode(self, vector, tokenizer):
        sep = vector.index(tokenizer.sep_token_id)
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
            "attention_mask": torch.tensor([1] * self.lengths[idx] + [0] * (self.max_seq_len - self.lengths[idx])) ,
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
            "next_sentence_label": self.next_sentence_label[idx],
        }
        return item

if __name__ == "__main__":
    bertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    print("loading dataset")
    train_dataset = BERTDataset('./test.tsv', bertTokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print("loading modle")
    vocab_size = len(bertTokenizer) + len(train_dataset.vid_vocab)
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(vocab_size)
    
    
    print("testing training")
    optimizer = optim.Adam(model.parameters(), 0.001)
    model = model.train()
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    scheduler = get_linear_schedule_with_warmup(optimizer, 4, 4)
    epochs = 2
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            next_sentence_label = batch["next_sentence_label"]
            
            prediction_logits, seq_relationship_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            optimizer.zero_grad()
            masked_lm_loss = loss_fn(prediction_logits.view(-1, vocab_size), labels.view(-1))
            next_sentence_loss = loss_fn(seq_relationship_logits.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            print("masked_lm_loss:", torch.exp(masked_lm_loss).item())
            print("next_sentence_loss:", torch.exp(next_sentence_loss).item())