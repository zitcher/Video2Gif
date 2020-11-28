import torch as th
from s3dg import S3D
from pathlib import Path
from PIL import Image, ImageSequence, ImageFile
import requests
import numpy as np
from scipy import ndimage

dict_path = "s3d_dict.npy"
weight_path = "s3d_howto100m.pth"
train_path = "../TGIF-Release-master/data/splits/train.txt"
tmp_path ="tmp3.gif"
save_path = "embeddings3/"
text_path = "correspondences3.txt"

ImageFile.LOAD_TRUNCATED_IMAGES = True

def file_to_embeddings(dict_path, weight_path, train_path, tmp_path, save_path):
    dict_path = Path(dict_path)
    weight_path = Path(weight_path)
    train_path = Path(train_path)
    tmp_path = Path(tmp_path)
    net = S3D(dict_path, 512)
    net.load_state_dict(th.load(weight_path))
    net.eval()
    with th.no_grad():
      net = net.double()
      with open(train_path) as f:
          urls = f.read().splitlines()

      embed = np.zeros((5000, 512))
      cnt = 0
      size = len(urls)
  
      # Save counter and file name for processing later
      txt = open(text_path, 'w')
      print(size)
      for i in range(3*size//5, 4*(size//5)):
          print(i)
          try:
            with open(tmp_path, 'wb') as f:
                f.write(requests.get(urls[i]).content)

            gif = Image.open(tmp_path)
          
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

            for j in range((frames.shape[2] // 32)+1):
              inp = frames[:,:,j*32:min((j+1)*32, frames.shape[2]),:,:]
              #print(inp.shape)
              if inp.shape[2] != 0:
                txt.write(urls[i] + " " + str(cnt))
                res = net(inp)['video_embedding'].detach().numpy()
                embed[cnt % 5000] = res
                cnt += 1
              if cnt % 5000 == 0 and cnt > 0:
                save = Path(save_path + str(i) + ".npy")
                with open(save, "wb") as f:
                    np.save(f, embed)
                embed = np.zeros((5000, 512))


          except Exception as e:
            print(str(i) + 'had value error')

      txt.close()
      save = Path(save_path + str(i) + ".npy")
      with open(save, "wb") as f:
        np.save(f, embed)
        embed = np.zeros((5000, 512))


file_to_embeddings(dict_path, weight_path, train_path, tmp_path, save_path)
