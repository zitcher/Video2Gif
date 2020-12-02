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
        'https://38.media.tumblr.com/8fd8930ec4e3458c1e616d6002e6a53f/tumblr_nqq3ibcw841rvg72ao1_400.gif',
        'https://33.media.tumblr.com/898ee85e0a05685d9c0cb326f9fd2280/tumblr_noijlk8qnH1uvid27o1_500.gif',
        'https://38.media.tumblr.com/7c3b37e8b79918eddb6fefb01d814bfe/tumblr_nmwkmcsEvf1tyncywo1_500.gif'
    ]

    tmp_path = './tmp.gif'
    net = ResNet101(weights='imagenet', include_top=False)

    embeddings = []
    for i in range(len(urls)):
        with open(tmp_path, 'wb') as f:
            f.write(requests.get(urls[i]).content)
    
            frames = [np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1], frame.size[0], 3) for frame in ImageSequence.Iterator(Image.open(tmp_path))]
            step = math.ceil(len(frames) / 5)
            sampled_frames = np.array([frames[i] for i in range(step // 2, len(frames), step)])
            inp = tf.image.resize(preprocess_input(sampled_frames), [244, 244], antialias=True)
            inp = net.predict(inp)
            embeddings.append(inp.flatten())

    print(np.linalg.norm(embeddings[0] - embeddings[1]), np.linalg.norm(embeddings[0] - embeddings[2]))

            
        
        