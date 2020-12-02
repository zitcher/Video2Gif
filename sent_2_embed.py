import torch as th
from s3dg import S3D
from pathlib import Path
from PIL import Image, ImageSequence
import requests
import numpy as np
import random


link_sent = "./link_sentence.tsv"


def get_embed_from_url(url):
    dict_path = "./data/s3dg/s3d_dict.npy"
    weight_path = "./data/s3dg/s3d_howto100m.pth"
    tmp_path ="tmp.gif"
    weight_path = Path(weight_path)
    tmp_path = Path(tmp_path)
    net = S3D(dict_path, 512).double()
    net.load_state_dict(th.load(weight_path))
    net.eval()
    with th.no_grad():
        with open(tmp_path, 'wb') as f:
            f.write(requests.get(url).content)

        gif = Image.open(tmp_path)

        embeds = []
        try:
            frames = np.array([np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(gif)])
            gif.close()
            frames = (np.reshape(frames, (1, 3, frames.shape[0], frames.shape[2], frames.shape[1]))) / 255.0
            frames = th.from_numpy(frames).detach().double()
            if frames.shape[2] % 2 == 1:
                frames = frames[:, :, :-1, :, :]
            if frames.shape[3] % 2 == 1:
                frames = frames[:,:,:,:-1,:]
            if frames.shape[4] % 2 == 1:
                frames = frames[:,:,:,:,:-1]
            res = net(frames)['video_embedding'].detach().numpy()

            for j in range((frames.shape[2] // 32)+1):
                inp = frames[:,:,j*32:min((j+1)*32, frames.shape[2]),:,:]
                if inp.shape[2] != 0:
                    res = net(inp)['video_embedding'].detach().numpy()
                    embeds.append(res)
        except ValueError as e:
            print(str(e) + 'had value error')
        
        return embeds
def get_embed_dict():
    # Hardcode the saved embeddings.
    file_list = ['3069.npy', '6129.npy', '9201.npy', '12240.npy', '15272.npy', '15999.npy', '19118.npy', '22200.npy', '25245.npy', '28305.npy',
    '31398.npy', '31999.npy', '35010.npy', '38063.npy', '41239.npy', '44290.npy', '47333.npy', '47999.npy', '51003.npy', '54089.npy', '57194.npy',
    '60246.npy', '63304.npy', '63999.npy', '67064.npy', '70193.npy', '73325.npy', '76381.npy', '79403.npy', '79999.npy']
    arrs = [] 

    #This is where things can get memory heavy
    for i in range(len(file_list)):
        inp = np.load('vgg_embeddings/' + file_list[i])

        if file_list[i] in ['15999.npy', '31999.npy', '47999.npy', '63999.npy', '79999.npy']:
            for j in range(inp.shape[0]):
                if np.sum(inp[j]) == 0:
                    inp = inp[:j,:]
                    break
        arrs.append(inp)
    
    data = np.concatenate(arrs, axis=0)
    del arrs

    url_map = dict()
    ct = 0
    id_to_res = dict()
    for k in range(5):
        with open('correspondences' + str(k) + '.txt','r') as f:
            urls = f.read().split(' ')
        
            for j in range(len(urls)-1):
                if j == 0:
                    url_map[urls[j]] = [ct]
                    ct += 1
                else:
                    idx = urls[j].split('h')[0]
                    cur_url = urls[j][len(urls[j].split('h')[0]):]
                    if cur_url in url_map:
                        url_map[cur_url].append(ct)
                        id_to_res[int(idx)] = ct
                        ct += 1
                    else:
                        url_map[cur_url] = [ct]
                        ct += 1
                        id_to_res[int(idx)] = ct

    return url_map, data

def sanity_check(url_map, data):
    k,v= random.choice(list(url_map.items()))
    embeds = get_embed_from_url(k)
    print(k)
    print(len(v))
    print(len(embeds))
    for index in range(len(v)):
        print(np.linalg.norm(data[v[index]]-embeds[index]), np.linalg.norm(data[v[index] +  1]-embeds[index]))

if __name__ == "__main__":
    embed_dict, data = get_embed_dict()