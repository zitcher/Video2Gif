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
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_video(path):
    if not os.path.exists(path):
        print("DOESN'T EXIST", path)
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    skip = 6
    buf = np.empty((frameCount // skip  + 1, frameHeight, frameWidth, 3), np.dtype('float32'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        if fc % skip != 0:
            fc += 1
            continue
        ret, frame = cap.read()
        if not ret:
            break
        
        buf[fc // skip] = frame
        fc += 1

    ## all dims must be divisible by 2 for s3dg
    if frameHeight % 2 == 1:
        buf = buf[:,:-1,:,:]
    if frameWidth % 2 == 1:
        buf = buf[:,:,:-1,:]

    return buf / 255.0
    
def nearest_crt(embedding, crts, row_to_crts):
    dists = scipy.spatial.distance.cdist(embedding, crts)
    dists = np.squeeze(dists)
    amin = np.argmin(dists)
    return row_to_crts[amin]


def video_tokenizer(video, net, window, crts, row_to_crts):
    tokens = ''
    for i in range(0, video.shape[0], window):
        vid_window = video[i:i+window]
        if vid_window.shape[0] < 10:
            continue
        
        ## all dims must be divisible by 2 for s3dg
        if vid_window.shape[0] % 2 == 1:
            vid_window=vid_window[:-1,:,:,:]

        # transform (frameCount, frameHeight, frameWidth, channels) to
        # (channels, frameCount, frameHeight, frameWidth)
        reindex = np.moveaxis(vid_window, 3, 0)
        single_batch = torch.tensor(reindex).unsqueeze(0).double().to(device)
        res = net(single_batch)['video_embedding'].detach().cpu().numpy()

        tokens += " v" + str(nearest_crt(res, crts, row_to_crts))
    return tokens.strip()
    

if __name__ == "__main__":
    print(device)
    dict_path = "./data/s3dg/s3d_dict.npy"
    weight_path = "./data/s3dg/s3d_howto100m.pth"
    video_path = "./data/yahoo/videos/half_videos"
    save_path = "youtube.txt"
    window = 16
    youtube_dataset = {'sequence': [], 'label': []}
    
    crts_dict = np.load('centers.npy', allow_pickle=True).item()
    crts = np.zeros((len(crts_dict), 512))
    row_to_crts = dict()

    index = 0
    for key, val in crts_dict.items():
        crts[index] = val
        row_to_crts[index] = key
        index +=1

    net = S3D(dict_path, 512).to(device)
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    with torch.no_grad():
        net = net.double()
        files = os.listdir(video_path)
        for dir in tqdm(files):
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

            tokens_pos = video_tokenizer(pos_video, net, window, crts, row_to_crts)
            print("pos", str(title) + ' [SEP] ' + tokens_pos)
            youtube_dataset['sequence'].append(str(title) + ' [SEP] ' + tokens_pos)
            youtube_dataset['label'].append(1)


            tokens_neg = video_tokenizer(neg_video, net, window, crts, row_to_crts)
            print("neg", str(title) + ' [SEP] ' + tokens_neg)
            youtube_dataset['sequence'].append(str(title) + ' [SEP] ' + tokens_neg)
            youtube_dataset['label'].append(0)

    datset = pd.DataFrame(data=youtube_dataset)
    dataset.to_csv('./youtube_dataset.csv')
