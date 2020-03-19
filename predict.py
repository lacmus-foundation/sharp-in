from tensorflow.keras.optimizers import Adam
import cv2
import os
from pathlib import Path
import numpy as np
from modules.u_net import define_unet

gpath = Path('')
imgs_folder = 'MyImages'

img_rows            = 512
img_cols            = 512   
weights_path        = Path(gpath, 'weights.h5')

unet = define_unet(img_rows, img_cols, optimizer=Adam(learning_rate=1e-4))

unet.load_weights(str(weights_path))

for filename in os.listdir(Path(gpath, imgs_folder)):
    if filename.endswith('.jpg'):
        
        X = cv2.imread(str(Path(gpath, imgs_folder, filename)))
        X = np.expand_dims(X,0)
        
        y_pred = unet.predict(X/255.)
        
        cv2.imwrite(str(Path(gpath, imgs_folder, filename[:-4]+'pred'+'.jpg')), (y_pred[0,]*255.).astype('uint8'))       
     