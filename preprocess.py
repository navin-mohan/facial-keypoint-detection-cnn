import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import cv2
import pandas as pd


df = pd.read_csv('./dataset/train.csv')

img = np.array([int(x) for x in df['Image'][0].split()],dtype=int)


print(img.shape)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

