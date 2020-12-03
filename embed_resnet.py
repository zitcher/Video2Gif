import torch as th
import tensorflow as tf
#from s3dg import S3D
from keras.preprocessing import image
from keras.applications.resnet import ResNet101
from keras.applications.resnet import preprocess_input


from pathlib import Path
from PIL import Image, ImageSequence, ImageFile
import requests
import numpy as np
from scipy import ndimage
import sys
import math

dict_path = "s3d_dict.npy"
weight_path = "s3d_howto100m.pth"
train_path = "../TGIF-Release-master/data/splits/train.txt"
#tmp_path ="tmp3.gif"
#save_path = "embeddings_vgg/"
#text_path = "vgg_correspondences3.txt"

ImageFile.LOAD_TRUNCATED_IMAGES = True

def file_to_embeddings(section, dict_path, weight_path, train_path):
    #dict_path = Path(dict_path)
    #weight_path = Path(weight_path)
    train_path = Path(train_path)
    tmp_path = Path('tmp' + str(section) + '.gif')
    save_path = 'resnet_embeddings/'
    text_path = Path('resnet_correspondences' + str(section) + '.txt')
    net = ResNet101(weights='imagenet', include_top=False, pooling='avg')
    #net.load_state_dict(th.load(weight_path))
    #net.eval()
    #with th.no_grad():
      #net = net.double()
    with open(train_path) as f:
        urls = f.read().splitlines()

    embed = np.zeros((5000, 2048))
    cnt = 0
    size = len(urls)
  
    # Save counter and file name for processing later
    txt = open(text_path, 'w')
    print(size)
    for i in range(section*size//5, (section+1)*(size//5)):
        print(i, flush=True)
        try:
          with open(tmp_path, 'wb') as f:
              f.write(requests.get(urls[i]).content)

          gif = Image.open(tmp_path)

          frames = [np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(Image.open(tmp_path))]
          step = math.ceil(len(frames) / 3)
          sampled_frames = np.array([frames[i] for i in range(step // 2, len(frames), step)])
          inp = tf.image.resize(preprocess_input(sampled_frames), [244, 244], antialias=True)
          inp = net.predict(inp)
          res = np.mean(inp, axis=0)
          embed[cnt % 5000] = res
          txt.write(urls[i] + " " + str(cnt))
          cnt += 1

          if cnt % 5000 == 0 and cnt > 0:
            save = Path(save_path + str(i) + ".npy")
            with open(save, "wb") as f:
                np.save(f, embed)
            embed = np.zeros((5000, 2048))


        except Exception as e:
          print(e)

    txt.close()
    save = Path(save_path + str(i) + ".npy")
    with open(save, "wb") as f:
      np.save(f, embed)
      embed = np.zeros((5000, 2048))

if __name__=="__main__":
  file_to_embeddings(int(sys.argv[1]), dict_path, weight_path, train_path)
