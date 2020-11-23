import torch as th
from s3dg import S3D
from pathlib import Path
from PIL import Image, ImageSequence
import requests
import numpy as np
from scipy import ndimage
import os

dict_path = "./data/s3dg/s3d_dict.npy"
weight_path = "./data/s3dg/s3d_howto100m.pth"
video_path = "./data/yahoo/videos/half_videos"
save_path = "youtube.txt"

if __name__ == "__main__":
    crts = np.load('centers.npy', allow_pickle=True)
    net = S3D(dict_path, 512)
    net.load_state_dict(th.load(weight_path))
    net.eval()
    with th.no_grad():
        net = net.double()
        files = os.listdir(video_path)
        for dir in files:
            dir_path = os.path.join(video_path, dir)
            if not os.path.isdir(os.path.join(video_path, dir)):
                continue
            
            metadata_path = os.path.join(dir_path, 'metadata.txt')
            pos_path = os.path.join(dir_path, 'pos.mp4')
            neg_path = os.path.join(dir_path, 'neg.mp4')
            df = pd.read_csv(metadata_path)
            title = df['title'][0]

            if not os.path.exists(neg_path):
                continue
                
            

