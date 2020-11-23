import torch
from s3dg import S3D
from pathlib import Path
from PIL import Image, ImageSequence
import requests
import numpy as np
from scipy import ndimage
import os
import cv2
import pandas as pd
import scipy.spatial

def read_video(path):
    if not os.path.exists(path):
        print("DOESN'T EXIST", path)
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount // 3  + 1, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        if fc % 3 != 0:
            fc += 1
            continue
        ret, frame = cap.read()
        if not ret:
            break
        
        buf[fc // 3] = frame
        fc += 1
    
    return buf
    
def nearest_crt(embedding, crts):
    print(crts.shape, embedding.shape)
    
    dists = scipy.spatial.distance.cdist(np.expand_dims(embedding, 0), crts)
    dists = np.squeeze(dists)

    print(dists.shape)


    
    return np.argmin(dists)


def video_tokenizer(video, net, window, crts):
    tokens = ''
    for i in range(0, video.shape[0], window):
        vid_window = video[i:i+window]
        # transform (frameCount, frameHeight, frameWidth, channels) to
        # (channels, frameCount, frameHeight, frameWidth)
        reindex = np.moveaxis(vid_window, 3, 0)
        single_batch = torch.tensor(reindex).unsqueeze(0).double()
        res = net(single_batch)['video_embedding'].detach().numpy()
        tokens += " v" + str(nearest_crt(res, crts))
    return tokens.strip()
    

if __name__ == "__main__":
    dict_path = "./data/s3dg/s3d_dict.npy"
    weight_path = "./data/s3dg/s3d_howto100m.pth"
    video_path = "./data/yahoo/videos/half_videos"
    save_path = "youtube.txt"
    youtube_dataset = {'sequence': [], 'label': []}
    
    crts_dict = np.load('centers.npy', allow_pickle=True).item()

    crts = np.zeros((len(crts_dict), 512))

    for key, val in crts_dict.items():
        crts[key] = val

    net = S3D(dict_path, 512)
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    with torch.no_grad():
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

            pos_video = read_video(pos_path)
            neg_video = read_video(neg_path)

            tokens_pos = video_tokenizer(pos_video, net, 32, crts)
            youtube_dataset['sequence'].append(title + ' [SEP] ' + tokens_pos)
            youtube_dataset['label'].append(1)


            tokens_ned = video_tokenizer(neg_video, net, 32, crts)
            youtube_dataset['sequence'].append(title + ' [SEP] ' + tokens_ned)
            youtube_dataset['label'].append(2)

            print(youtube_dataset)

    datset = pd.DataFrame(data=youtube_dataset)
    dataset.to_csv('./youtube_dataset.csv')
