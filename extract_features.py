from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pandas as pd
import numpy as np
import h5py

model = VGG16(  weights="imagenet",
                include_top=False
             )


def get_features(img_paths):
    imgs = [image.load_img(img_path, target_size=(224, 224)) for img_path in img_paths ]
    x = np.array([image.img_to_array(img) for img in imgs])
    # x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return np.array([f.flatten() for f in features])


l = pd.read_csv("./dataset/train.csv").shape[0]

features = None

l = 22

batch_size = 10

for i in range(0,l,batch_size):
    f = get_features(["./dataset/train/{}.jpg".format(index) for index in range(i,i+batch_size) if index < l ])
    if features is None:
        features = f
    else:
        features = np.concatenate((features,f))
    print("{0} of {1} completed..".format(i+1,l))


assert features is not None

h5f = h5py.File('features.h5','w')
h5f.create_dataset('train_features',data=features)
h5f.close()

