import torch as th
from s3dg import S3D
from pathlib import Path
from PIL import Image, ImageSequence
import requests
import numpy as np
from scipy import ndimage
import random
import imageio

out_path = "out.gif"
dict_path = "res_tks_to_gifs.npy"
tmp_path = "tmp.gif"

def build_gif(tokens):
  mapping = np.load(dict_path, allow_pickle=True)

  im_list = []

  for t in tokens:
    cluster = mapping.item().get(t)
    # Cluster should be a tuple of (url, idx)
    url, idx = random.choice(cluster)
    with open(tmp_path, 'wb') as f:
      f.write(requests.get(url).content)

    gif = Image.open(tmp_path)
    frames = np.array([np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(gif)])
    for i in range(idx*32, min(frames.shape[0], (idx+1)*32)):
      im = Image.fromarray((np.squeeze(frames[i])))
      im_list.append(im)
  imageio.mimsave('out.gif', im_list)
      


build_gif([0, 167, 1899])
