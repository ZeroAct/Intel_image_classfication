import json, os, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, optimizers, losses, applications

# set random seed 123
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

# dataset
train_datagen = ImageDataGenerator(
    rotation_range=20,
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
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    'dataset/seg_test',
    target_size=(150, 150),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# save class info
labels = {}
for k, v in validation_generator.class_indices.items():
    labels[v] = k
    
js = json.dumps(labels)
f = open("labels.json","w")
f.write(js)
f.close()

# train setting
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


# model
model = models.Sequential()
resnet = applications.ResNet50(include_top=False, input_shape=(150, 150, 3))

for layer in resnet.layers:
    layer.trainable = False

model.add(resnet)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer=optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['acc'])

# training
hist = model.fit_generator(generator=train_generator,
                           steps_per_epoch=STEP_SIZE_TRAIN,
                           validation_data=validation_generator,
                           validation_steps=STEP_SIZE_VALID,
                           epochs=100, callbacks=[cp_callback, rd_callback, es_callback])

# visualize
name = "final"

tmp = hist.history.copy()
del tmp['lr']
js = json.dumps(tmp)
f = open(name + ".json","w")
f.write(js)
f.close()


with open(name+'.json', 'r') as dic:
    hist = json.load(dic)
    
epochs = [str(e) for e in np.arange(len(hist['acc']))]

plt.title(name+"_acc")
plt.plot(epochs, hist['acc'], label='train_acc')
plt.plot(epochs, hist['val_acc'], label='val_acc')
plt.legend()

plt.savefig(name+"_acc.png")
plt.close()

plt.title(name+"_loss")
plt.plot(epochs, hist['loss'], label='train_loss')
plt.plot(epochs, hist['val_loss'], label='val_loss')
plt.legend()

plt.savefig(name+"_loss.png")
plt.close()
