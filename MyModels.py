from tensorflow.keras import layers, models, optimizers, losses, applications

def CreateModel():
    model = models.Sequential()
    model.add(applications.ResNet50(include_top=False, input_shape=(150, 150, 3)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(6, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    return model

def CreateModelf():
    model = models.Sequential()
    resnet = applications.ResNet50(include_top=False, input_shape=(150, 150, 3))
    
    for layer in resnet.layers:
        layer.trainable = False
    
    model.add(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(6, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    return model

def CreateModeld():
    model = models.Sequential()
    resnet = applications.ResNet50(include_top=False, input_shape=(150, 150, 3))
    
    for layer in resnet.layers:
        layer.trainable = False
    
    model.add(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(6, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    return model

def CreateModeldd():
    model = models.Sequential()
    resnet = applications.ResNet50(include_top=False, input_shape=(150, 150, 3))
    
    for layer in resnet.layers:
        layer.trainable = False
    
    model.add(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    return model

def CreateModelnd():
    model = models.Sequential()
    resnet = applications.ResNet50(include_top=False, input_shape=(150, 150, 3))
    
    for layer in resnet.layers:
        layer.trainable = False
    
    model.add(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    return model

def CreateModeld3d():
    model = models.Sequential()
    resnet = applications.ResNet50(include_top=False, input_shape=(150, 150, 3))
    
    for layer in resnet.layers:
        layer.trainable = False
    
    model.add(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(6, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    return model