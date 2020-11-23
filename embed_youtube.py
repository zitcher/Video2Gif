import torch as th
from s3dg import S3D
from pathlib import Path
from PIL import Image, ImageSequence
import requests
import numpy as np
from scipy import ndimage

if __name__ == "__main__":
    crts = np.load('centers.npy', allow_pickle=True)
    