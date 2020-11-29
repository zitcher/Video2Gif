from torch.utils.data import Dataset, DataLoader
import torch
from sentence_transformers import SentenceTransformer
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
from sent_2_embed import get_embed_dict
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPT2MappingModel(torch.nn.Module):
    def __init__(self):
        super(GPT2MappingModel, self).__init__()
        self.hidden = 768 # https://huggingface.co/transformers/pretrained_models.html
        self.out = 512
        self.l1 = nn.Linear(self.hidden, self.out)
    
    def forward(self, x):
        return self.l1(x)

class GPT2Dataset(Dataset):
    def __init__(self, sentence_embeds, gif_embeds):
        self.sembed = []
        self.gembed = []
        for i in range(sentence_embeds.shape[0]):
            self.sembed.append(torch.tensor(sentence_embeds[i]).float())
            self.gembed.append(torch.tensor(gif_embeds[i]).float())
    
    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        return len(self.sembed)

    def __getitem__(self, idx):
        """
        :return: dictionary of the data
        """
        item = {
            "sentence_embeds": self.sembed[idx],
            "gif_embeds": self.gembed[idx],
        }
        return item


if __name__ == "__main__":
    if not os.path.exists("./gifs_embeds.out") or not os.path.exists("./sentence_embeds.out"):
        print("generating embeddings")
        embed_dict, embedding_array = get_embed_dict()
        print("loading gpt2")
        transformer = SentenceTransformer('bert-base-nli-mean-tokens').to(device)

        link_sent = "./link_sentence.tsv"

        with open(link_sent) as f:
            raw = f.read()
        
        lines = raw.split('\n')
        gifs_embeds = []
        sentence_embeds = []

        for line in tqdm(lines):
            line = line.split('\t')
            link = line[0]
            sentence = line[1]

            if link not in embed_dict:
                continue

            embed_indices = embed_dict[link]
            sections = []
            for idx in embed_indices:
                sections.append(embedding_array[idx])

            mean = np.mean(np.array(sections), axis=0) if len(sections) > 1 else sections[0]
            gifs_embeds.append(mean)
            
            sentence_embeddings = transformer.encode([sentence])
            sentence_embeds.append(sentence_embeddings[0])

        np.savetxt('./gifs_embeds.out', gifs_embeds, delimiter=',')
        np.savetxt('./sentence_embeds.out', sentence_embeds, delimiter=',')
    else:
        print("loading data")
    gifs_embeds = np.loadtxt('./gifs_embeds.out', delimiter=',')
    sentence_embeds = np.loadtxt('./sentence_embeds.out', delimiter=',')

    print(gifs_embeds.shape)
    print(sentence_embeds.shape)

    dataset = GPT2Dataset(sentence_embeds, gifs_embeds)

    model = GPT2MappingModel()
    model = model.to(device)

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    epochs = 2
    optimizer = optim.Adam(model.parameters(), 0.001)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(train_loader):
            input_embeds = batch["sentence_embeds"].to(device)
            labels = batch["gif_embeds"].to(device)
            out = model(input_embeds)

            optimizer.zero_grad()
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("epoch", epoch, "loss:", np.mean(losses))
        torch.save(model.state_dict(), './checkpoints/{}.cpt'.format(epoch))