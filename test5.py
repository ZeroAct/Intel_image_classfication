
import json, os, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, optimizers, losses, applications

from MyModels import CreateModelnd

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    preprocessing_function=applications.resnet50.preprocess_input)

test_datagen = ImageDataGenerator(
    preprocessing_function=applications.resnet50.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'dataset/seg_train',
    target_size=(150, 150),
    color_mode='rgb',
    batch_size=16,
    class_mode='categorical',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    'dataset/seg_test',
    target_size=(150, 150),
    color_mode='rgb',
    batch_size=16,
    class_mode='categorical',
    shuffle=False)

labels = {}
for k, v in validation_generator.class_indices.items():
    labels[v] = k
    
js = json.dumps(labels)
f = open("labels.json","w")
f.write(js)
f.close()

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
 
checkpoint_history = os.listdir("training/")
os.mkdir("training/"+str(len(checkpoint_history)))
checkpoint_path = "training/"+str(len(checkpoint_history))+"/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=5)

rd_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                   patience=3, min_lr=0.00001)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model = CreateModelnd()

hist = model.fit_generator(generator=train_generator,
                           steps_per_epoch=STEP_SIZE_TRAIN,
                           validation_data=validation_generator,
                           validation_steps=STEP_SIZE_VALID,
                           epochs=100, callbacks=[cp_callback, rd_callback, es_callback])

tmp = hist.history.copy()
del tmp['lr']
js = json.dumps(tmp)
f = open("history_preprocess_freeze_rd_aug_nodrop_dense.json","w")
f.write(js)
f.close()
