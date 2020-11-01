from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer, BertForPreTraining
import random
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from torch import nn, optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def replaceEl(lst, oldval, newval):
    newlst = []
    for item in lst:
        if item == oldval:
            newlst.append(newval)
        else:
            newlst.append(item)
    return newlst

class BERTDataset(Dataset):
    def __init__(self, input_file, tokenizer):
        """
        Expects a file of where each lines is a sentence of: word1, word2, ... wordk [SEP] vidtoken1, vidtoken2, vidtokenk
        """
        vid_vocab = dict()

        # read the input file line by line and put the lines in a list.
        with open(input_file) as f:
            self.raw = f.read()

        sentences = self.raw.split('\n')
        

        # list of (query, response)
        new_tokens = []
        for line in sentences:
            sentence_vids = line.split('[SEP]')
            vids = sentence_vids[1].strip().split(' ')

            for vid in vids:
                if vid not in vid_vocab:
                    vid_vocab[vid] = 1
                    new_tokens.append(vid)

        special_tokens_dict = {'additional_special_tokens': new_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)

        # get dataset in a random order, impt for next sent prediction construction
        # since we use the next sentence in the list as the example of an incorrect next sentence prediction
        random.shuffle(sentences)

        self.input_ids = []
        self.lengths = []
        self.max_seq_len = 0
        self.token_type_ids = [] # 0 for sent A, 1 for sent B
        self.labels = [] # -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size]
        self.next_sentence_label = [] # Indices should be in [0, 1], 0 indicates sequence B is a continuation of sequence A
        for i in range(len(sentences)):
            # add a correct next sentence example
            line = sentences[i]
            linesplit = line.split('[SEP]')
            
            
            vector = self.vectorize(line, tokenizer)
            self.input_ids.append(torch.tensor(vector))
            self.lengths.append(len(vector))
            self.max_seq_len = max(self.max_seq_len, len(vector))
            self.token_type_ids.append(torch.tensor(
                ([0] * (len(self.vectorize(linesplit[0].strip(), tokenizer)))) + 
                ([1] * (len(self.vectorize(linesplit[1].strip(), tokenizer)) - 1))
            ))

            vector = replaceEl(vector, tokenizer.cls_token_id, -100)
            vector = replaceEl(vector, tokenizer.sep_token_id, -100)
            self.labels.append(torch.tensor(vector))
            self.next_sentence_label.append(torch.tensor([0]))

            # add a incorrect next sentence example
            nextline = None
            if i < len(sentences) - 1:
                nextline = sentences[i + 1]
            else:
                nextline = sentences[0]

            nextlinesplit = nextline.split('[SEP]')

            inccorrect_next_sent = linesplit[0].strip() + ' [SEP] ' + nextlinesplit[1].strip()
            vector = self.vectorize(inccorrect_next_sent, tokenizer)
            self.input_ids.append(torch.tensor(vector))
            self.lengths.append(len(vector))
            self.max_seq_len = max(self.max_seq_len, len(vector))
            self.token_type_ids.append(torch.tensor(
                ([0] * (len(self.vectorize(linesplit[0].strip(), tokenizer)))) + 
                ([1] * (len(self.vectorize(nextlinesplit[1].strip(), tokenizer)) - 1))
            ))
            vector = replaceEl(vector, tokenizer.cls_token_id, -100)
            vector = replaceEl(vector, tokenizer.sep_token_id, -100)
            self.labels.append(torch.tensor(vector))
            self.next_sentence_label.append(torch.tensor([1]))

        self.input_ids = pad_sequence(self.input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        self.token_type_ids = pad_sequence(self.token_type_ids, batch_first=True, padding_value=1)
        self.labels = pad_sequence(self.labels, batch_first=True, padding_value=-100)

        print(self.input_ids.size())
        print(self.token_type_ids.size())
        print(self.labels.size())
        
    def vectorize(self, line, tokenizer):
        return tokenizer.encode(line)

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
    bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("loading dataset")
    train_dataset = BERTDataset('./test.tsv', bertTokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print("loading modle")
    vocab_size = len(bertTokenizer)
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(vocab_size)
    
    
    print("testing training")
    optimizer = optim.Adam(model.parameters(), 0.01)
    model = model.train()
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    scheduler = get_linear_schedule_with_warmup(optimizer, 2, 2)
    epochs = 2
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            next_sentence_label = batch["next_sentence_label"]

            # print(bertTokenizer.decode(input_ids[0]), next_sentence_label[0])
            # print(bertTokenizer.decode(labels[0]))

            # print("RAW")
            # print(input_ids[0])
            # print(attention_mask[0])
            # print(token_type_ids[0])
            # print(labels[0])

            # print("END")
            
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

            perplexity = torch.exp(total_loss).item()
            print("perplexity:", perplexity)