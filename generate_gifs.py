from preprocess import load_obj
import os
import torch
from build_gifs import build_gif
from sent_2_embed import get_embed_dict
from mapping_model import GPT2MappingModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# basepath = '/users/zhoffman/Video2Gif'
basepath = '.'

def decode(vector, tokenizer, vid_vocab, vocab_vid):
    gifs = ''
    for vid in vector:
        if vid in vocab_vid:
            gifs += ' ' + vocab_vid[vid]
        else:
            gifs += tokenizer.decode([vid])

    return gifs
def decode_gif(vector, vocab_vid):
    gifs = ''
    for vid in vector:
        gifs += vocab_vid[vid] + ' '
    return gifs.strip()

def get_max_token(logits, vocab_vid):
    best = None
    prob = None
    word = None
    for vocab, vid in vocab_vid.items():
        if vid == '[SEP]':
            continue
        if best is None or logits[vocab] > prob:
            best = vocab
            prob = logits[vocab]
            word = vid
    return best, word

def sentence_to_tokens(sentence, model, tokenizer, vid_vocab, vocab_vid):
    global device

    model = model.to(device)
    model_in = torch.tensor([tokenizer.encode(sentence.strip()) + [vid_vocab['[SEP]']]]).to(device)

    current_sentence = model_in.tolist()[0]
    gif = []

    for i in range(1):
        out = model(
            input_ids=model_in
        )

        # (batch_size, sequence_length, config.vocab_size)
        logits = out.logits
        next_word, vid = get_max_token(logits[0, -1, :].tolist(), vocab_vid)
        gif.append(int(vid[1:]))
        current_sentence.append(next_word)
        model_in = torch.tensor([current_sentence]).to(device)

        print("logits", decode(torch.argmax(logits, dim=2).tolist()[0], tokenizer, vid_vocab, vocab_vid))
        print("current", decode(current_sentence, tokenizer, vid_vocab, vocab_vid))
    print(gif)
    return gif

def generate_gif_gpt2_nexttok(sentence):
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
    print(len(vid_vocab))
    vocab_size = len(gpt2Tokenizer) + len(vid_vocab)
    model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True)
    model.resize_token_embeddings(vocab_size)
    model.load_state_dict(torch.load(basepath + '/checkpoints/{}.cpt'.format('100')))

    line = sentence_to_tokens(
        "Where are the pandas", 
        model, 
        gpt2Tokenizer, 
        vid_vocab, 
        vocab_vid
    )
    build_gif(line)

def generate_gif_gpt2_mapping(sentence):
    with torch.no_grad():
        print("loading models")
        embed_dict, embedding_array = get_embed_dict()
        transformer = SentenceTransformer('bert-base-nli-mean-tokens').to(device)

        print("generating embedding")    
        encode = transformer.encode([sentence])
        encode = torch.tensor(encode).to(device)
        print(encode.size())

        model = GPT2MappingModel()
        model.load_state_dict(torch.load('./checkpoints/{}.cpt'.format(0)))
        model = model.to(device)
        model.eval()
        vid_embed = model(encode)
        vid_embed = np.array(vid_embed.tolist()[0])

        print("finding best match")
        best = None
        best_dst = None
        for url, embed_indices in tqdm(embed_dict.items()):
            sections = []
            for idx in embed_indices:
                sections.append(embedding_array[idx])
            
            mean = np.mean(np.array(sections), axis=0) if len(sections) > 1 else sections[0]

            dst = np.mean(np.square(mean-vid_embed))

            if best is None or dst < best_dst:
                best = url
                best_dst = dst
        print("match score", dst, url)
        with open('./sentence.gif', 'wb') as f:
            f.write(requests.get(best).content)
    

if __name__ == "__main__":
    generate_gif_gpt2_mapping("woman on the phone with her mom")