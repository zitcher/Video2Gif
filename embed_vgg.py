import torch as th
#from s3dg import S3D
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


from pathlib import Path
from PIL import Image, ImageSequence, ImageFile
import requests
import numpy as np
from scipy import ndimage

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
    save_path = 'vgg_embeddings2/'
    text_path = Path('vgg_correspondences' + str(section) + '.txt')
    net = VGG16(weights='imagenet', include_top=False) #S3D(dict_path, 512)
    #net.load_state_dict(th.load(weight_path))
    #net.eval()
    #with th.no_grad():
      #net = net.double()
    with open(train_path) as f:
        urls = f.read().splitlines()

    embed = np.zeros((5000, 512*7*7))
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
          
          frames = np.array([np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(gif)])
          gif.close()
          #frames = (np.reshape(frames, (1, 3, frames.shape[0], frames.shape[1], frames.shape[2]))) #/ 255.0
          #print(frames.shape)
          #frames = th.from_numpy(frames).detach().double()
          #if frames.shape[2] % 2 == 1:
          #  frames = frames[:, :, :-1, :, :]
          #if frames.shape[3] % 2 == 1:
          #  frames = frames[:,:,:,:-1,:]
          #if frames.shape[4] % 2 == 1:
          #  frames = frames[:,:,:,:,:-1]

          for j in range((frames.shape[0] // 32)+1):
            #inp = frames[:,:,j*32:min((j+1)*32, frames.shape[2]),:,:]
            inp = frames[j*32:min((j+1)*32, frames.shape[0]), :, :, :]
            ids = np.random.randint(min((j+1)*32, inp.shape[0]), size=5)
            if inp.shape[0] >= 4:
              inp = inp[ids, :,:,:]
              inp = np.squeeze(inp)
              print(inp.shape)
              #inp = np.reshape(inp, (inp.shape[1],inp.shape[2], inp.shape[3], inp.shape[0]))
              res_inp = np.zeros((5,224,224,3))
              for m in range(5):
                t = inp[m,:,:,:]
                #print(t.shape)
                t = Image.fromarray(np.uint8(t))
                t = t.resize((224,224))
                # For debugging
                # t.show()
                res_inp[m] = np.array(t)
              inp = preprocess_input(res_inp)
              if inp.shape[0] != 0:
                inp = net.predict(inp) 
                # Take average over the output features to save space
                inp = np.mean(inp, axis=0)
                print(inp.shape)
                res = inp.flatten()
                embed[cnt % 5000] = res
                txt.write(urls[i] + " " + str(cnt))
                cnt += 1
              if cnt % 5000 == 0 and cnt > 0:
                save = Path(save_path + str(i) + ".npy")
                with open(save, "wb") as f:
                    np.save(f, embed)
                embed = np.zeros((5000, 512*7*7))


        except Exception as e:
          print(e)

    txt.close()
    save = Path(save_path + str(i) + ".npy")
    with open(save, "wb") as f:
      np.save(f, embed)
      embed = np.zeros((5000, 512*7*7))


file_to_embeddings(0, dict_path, weight_path, train_path)
