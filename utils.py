
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, optimizers, losses
from CreateModel import CreateModel

model = CreateModel()

checkpoint_dir = "training/"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

with open('labels.json', 'r') as dic:
    labels = json.load(dic)

def ShowFeatureMap(layer_num, img):
    assert len(model.layers) > layer_num
    assert type(model.layers[layer_num]) == tf.python.keras.layers.convolutional.Conv2D
    
    for i in range(layer_num+1):
        img = model.layers[i](img)
        
    img = img[0].numpy()
    
    height = 8
    width = img.shape[-1] // height
    
    fig, axs = plt.subplots(height, width, figsize=(20, 20), )
    for h in range(height):
        for w in range(width):
            axs[h, w].imshow(img[:,:,h*width + w])
            axs[h, w].axis('off')
    return axs

def Predict(img_path):
    img = load_img("dataset/seg_pred/"+img_path,
                   color_mode='rgb',
                   target_size=(150, 150),
                   interpolation='cubic')
    img = img_to_array(img)/255.
    img = np.array([img])

    res = model.predict(img)[0]
    res = np.argmax(res)
    
    print(labels[str(res)])

if __name__ == "__main__":
    img_path = "3.jpg"
    img = load_img("dataset/seg_pred/"+img_path,
                   color_mode='rgb',
                   target_size=(150, 150),
                   interpolation='cubic')
    img = img_to_array(img)/255.
    img = np.array([img])
    
    res = ShowFeatureMap(0, img)
