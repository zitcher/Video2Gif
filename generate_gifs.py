device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sentence_to_tokens(sentence, model):
    global device
    model = model.to(device)

    model_in = sentence.strip() + ' [SEP]'
