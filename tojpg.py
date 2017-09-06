from scipy.misc import imsave
import pandas as pd
import numpy as np

df = pd.read_csv('./dataset/train.csv')


def to_np_array(row):
    return np.array([ int(x) for x in row['Image'].split() ],dtype=int).reshape((96,96))


l = df.shape[0]

for i,row in df.iterrows(): 
    img = to_np_array(row)
    imsave('./dataset/train/{}.jpg'.format(i),img)
    print("{0} of {1} completed...".format(i+1,l))

df = df.drop('Image',axis=1)

df.to_csv('./dataset/without_img.csv')