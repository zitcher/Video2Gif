from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizerFast, BertForPreTraining
import random
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from torch import nn, optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from preprocess import BERTDataset
load = False
loadepoch = 0
# basepath = '/users/zhoffman/Video2Gif'
basepath = '.'

if __name__ == "__main__":
    bertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    print("loading dataset")
    train_dataset = BERTDataset(basepath + '/training.tsv', bertTokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print("loading modle")
    vocab_size = len(bertTokenizer) + len(train_dataset.vid_vocab)
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(vocab_size)

    if load:
        epoch = 0
        model.load_state_dict(torch.load(basepath + '/checkpoints/{}.cpt'.format(loadepoch)))
    
    model = model.cuda()
    print("testing training")
    optimizer = optim.Adam(model.parameters(), 0.001)
    model = model.train()
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    scheduler = get_linear_schedule_with_warmup(optimizer, 4, 4)
    epochs = 20
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            token_type_ids = batch["token_type_ids"].cuda()
            labels = batch["labels"].cuda()
            next_sentence_label = batch["next_sentence_label"].cuda()
            
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
        torch.save(model.state_dict(), basepath + '/checkpoints/{}.cpt'.format(epoch))