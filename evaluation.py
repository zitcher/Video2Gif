import numpy as np
from tqdm import tqdm, trange
from nltk.stem.porter import PorterStemmer
from preprocess import load_obj
import os
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset(file):
    stemmer = PorterStemmer()

    with open(file) as f:
        text = f.read()
    
    sentences = text.split('\n')
    
    lines_vids = []
    vocab = dict()
    for line in sentences:
        sp = line.split('[SEP]')
        lines_vids.append((sp[0].strip(), sp[1].strip()))
        for word in sp[0].strip().split(' '):
            stem = stemmer.stem(word)
            if stem not in vocab:
                vocab[stem] = len(vocab)

    return lines_vids, vocab

def one_hot(sentence, vocab, stemmer):
    one_hot =  np.zeros(len(vocab))

    for word in sentence.split(' '):
        stem = stemmer.stem(word)
        if stem in vocab:
            one_hot[vocab[stem]] = 1

    return one_hot

# accuracy 0.005462457179890751
def NNEval():
    train_file = './res_training_final.tsv'
    val_file = './res_training_val.tsv'
    print("Loading datasets ")

    train_dataset, train_vocab = get_dataset(train_file)
    val_dataset, val_vocab = get_dataset(val_file)

    stemmer = PorterStemmer()

    print("loading hots")
    onehots = np.zeros((len(train_dataset), len(train_vocab)))
    for i in trange(len(train_dataset)):
        line, vid = train_dataset[i]
        thot = one_hot(line, train_vocab, stemmer)
        onehots[i] = thot

    print("Evaluating NN")
    correct = 0
    total = len(val_dataset)
    for vline, vvid in tqdm(val_dataset):
        vhot = one_hot(vline, train_vocab, stemmer)
        score_mat =  np.dot(onehots, vhot)
        best_row = np.argmax(score_mat)
        line = train_dataset[best_row][0]
        vid = train_dataset[best_row][1]
        if vid == vvid:
            correct += 1

    return correct / total

# top1 0.0085 
# top 10 0.04
# top 100 0.13
def GPT2Eval():
    val_file = './res_training_val.tsv'
    print("Loading datasets")

    val_dataset, val_vocab = get_dataset(val_file)

    print("Evaluating GPT2")
    correct = 0
    total = len(val_dataset)

    print("device", device)
    print("loading gpt2 tokenizer")
    gpt2Tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print("loading vid vocab")
    vid_vocab = None
    vocab_vid = None
    if os.path.exists("./data/vid_vocab.pkl") and os.path.exists("./data/vocab_vid.pkl"):
        vid_vocab = load_obj("vid_vocab")
        vocab_vid = load_obj("vocab_vid")
    else:
        print("failed to find vocab files")
        assert False

    print("loading model")
    vocab_size = len(gpt2Tokenizer) + len(vid_vocab)
    model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
    model.resize_token_embeddings(vocab_size)
    model.load_state_dict(torch.load('./checkpoints/{}.cpt'.format('25')))
    for sentence, vid in tqdm(val_dataset):
        model = model.to(device)
        model_in = torch.tensor([gpt2Tokenizer.encode(sentence.strip()) + [vid_vocab['[SEP]']]]).to(device)

        out = model(input_ids=model_in)
        vidids = torch.topk(out.logits, 1000, dim=2).indices[0, -1].tolist()

        if vid not in vid_vocab:
            total -= 1
            continue
        ind = vid_vocab[vid]
        if ind in vidids:
            correct += 1

        print(correct / total)

    return correct / total


if __name__ == "__main__":
    # baseline = NNEval()
    model = GPT2Eval()

    # print("model", model, "baseline", baseline)