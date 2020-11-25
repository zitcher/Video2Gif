from preprocess import load_obj
import os
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from build_gifs import build_gif

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# basepath = '/users/zhoffman/Video2Gif'
basepath = '.'

def decode(vector, tokenizer, vid_vocab, vocab_vid):
    sep = vector.index(vid_vocab['[SEP]'])
    gifs = ''
    for vid in vector[sep:len(vector)]:
        if vid in vocab_vid:
            gifs += ' ' + vocab_vid[vid]
        else:
            gifs += tokenizer.decode([vid])

    return tokenizer.decode(vector[:sep + 1]) + gifs

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

        print(decode(current_sentence, tokenizer, vid_vocab, vocab_vid))
    print(gif)
    return gif

if __name__ == "__main__":
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
        "A panda sits on a log", 
        model, 
        gpt2Tokenizer, 
        vid_vocab, 
        vocab_vid
    )
    build_gif(line)