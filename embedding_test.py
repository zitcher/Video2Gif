import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
import requests
from PIL import Image, ImageSequence
import numpy as np
import math 
from array2gif import write_gif

if __name__ == "__main__":
    urls = [
        'https://33.media.tumblr.com/ab7342b8cf218a50fc80b59896393b9e/tumblr_nq3ym5BfvM1uv0y2ro1_250.gif',
        'https://33.media.tumblr.com/b9e43f1f2905aa9d2b92a58591372b83/tumblr_nnuv5ro5eX1td4njbo1_400.gif',
        'https://33.media.tumblr.com/4ccfa9269be73b6ecf451e035707558d/tumblr_no7b56gmOQ1qa05h5o1_400.gif'
    ]

    tmp_path = './tmp.gif'
    net = ResNet101(weights='imagenet', include_top=False, pooling='avg')

    embeddings = []
    for i in range(len(urls)):
        with open(tmp_path, 'wb') as f:
            f.write(requests.get(urls[i]).content)
    
            frames = [np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(Image.open(tmp_path))]
            step = math.ceil(len(frames) / 3)
            sampled_frames = np.array([frames[i] for i in range(step // 2, len(frames), step)])
            inp = tf.image.resize(preprocess_input(sampled_frames), [244, 244], antialias=True)
            print(inp.shape)
            inp = net.predict(inp)
            print(inp.shape)
            embeddings.append(np.mean(inp, axis=0))

    print(np.linalg.norm(embeddings[0] - embeddings[1]), np.linalg.norm(embeddings[0] - embeddings[2]))