from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
from pathlib import Path
import numpy as np
from modules.batch_generator import BatchGenerator     
from modules.u_net import define_unet
from modules.cosine_lr import CosineLR 
from datetime import datetime


gpath = Path('')
dataset_path        = Path('E:\c6i512')       

test_trainval_split = 0.02
trainval_split      = 0.8

img_rows            = 512
img_cols            = 512   

# train-val split
indices = (np.array(os.listdir(Path(dataset_path, 'X'))))
np.random.shuffle(indices)
# indices = indices[:50]

trainval_indices = indices[int(test_trainval_split*len(indices)):]
test_indices = indices[:int(test_trainval_split*len(indices))]

train_indices = trainval_indices[:int(trainval_split*len(trainval_indices))]
val_indices = trainval_indices[int(trainval_split*len(trainval_indices)):]

print('Images:')
print('Train:%i, val:%i, test:%i, total:%i' % (len(train_indices), len(val_indices), len(test_indices), len(indices)))

unet = define_unet(img_rows, img_cols, optimizer=Adam(learning_rate=1e-4))

# training config
batch_size     = 8
n_epochs       = 10

# data generators
train_generator = BatchGenerator(dataset_path, train_indices, dim=(img_rows, img_cols), batch_size=batch_size)
val_generator   = BatchGenerator(dataset_path, val_indices,   dim=(img_rows, img_cols), batch_size=batch_size)

# callbacks
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
# lr_reduce  = ReduceLROnPlateau(monitor='loss', min_lr=0, cooldown=0, factor=0.2, patience=2, verbose=1, mode='min')
lr_reduce = CosineLR(min_lr=1e-12, max_lr=1e-4, steps_per_epoch=np.ceil(len(train_indices)/batch_size), lr_decay=0.9)

# train model
history = unet.fit_generator(generator=train_generator, validation_data=val_generator, 
                             epochs=n_epochs, callbacks=[lr_reduce, early_stop],                              
                             verbose=1, workers=0, use_multiprocessing=False)

r_name = (str(datetime.now())[:16]).replace(':','-')
unet.save_weights(str(Path(gpath, r_name+'.h5')))

    