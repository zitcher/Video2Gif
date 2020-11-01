import torch as th
from s3dg import S3D
from pathlib import Path
from PIL import Image, ImageSequence
import requests
import numpy as np
from scipy import ndimage

dict_path = "I:/s3d_dict.npy"
weight_path = "I:/s3d_howto100m.pth"
train_path = "I:/TGIF-Release-master/data/splits/train.txt"
tmp_path ="I:/tmp.gif"
save_path = "I:/embeddings/"

def file_to_embeddings(dict_path, weight_path, train_path, tmp_path, save_path):
    embed = None
    dict_path = Path(dict_path)
    weight_path = Path(weight_path)
    train_path = Path(train_path)
    tmp_path = Path(tmp_path)
    net = S3D(dict_path, 512)
    net.load_state_dict(th.load(weight_path))
    net = net.double()
    net = net.eval()

    with open(train_path) as f:
        urls = f.read().splitlines()

    for i in range(len(urls)):
        print(urls[i])
        with open(tmp_path, 'wb') as f:
            f.write(requests.get(urls[i]).content)

        gif = Image.open(tmp_path)
        frames = np.array([np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(gif)])
        frames = (np.reshape(frames, (1, 3, frames.shape[0], frames.shape[2], frames.shape[1]))) / 255.0
        frames = th.from_numpy(frames).double()
        if embed == None:
            embed = net(frames)['video_embedding'].detach().numpy()
        else:
            embed = np.stack(embed, net(frames).numpy())

        if i % 5000 == 0:
            save = Path(save_path + str(i) + ".npy")
            with open(save, "wb") as f:
                np.save(f, embed)
            embed = None

file_to_embeddings(dict_path, weight_path, train_path, tmp_path, save_path)