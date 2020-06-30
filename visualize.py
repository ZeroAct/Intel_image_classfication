import matplotlib.pyplot as plt
import json
import numpy as np

name = "history_preprocess_freeze_rd_aug"

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


