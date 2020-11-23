from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Convolution2D, MaxPooling2D, UpSampling2D
from pathlib import Path
from tensorflow.keras.optimizers import Adam
import cv2
import os
import time
import numpy as np
from core.config import get_config

crop_size = get_config().crop_size
    
def define_unet(img_rows, img_cols, optimizer):
    ''' Defines U-net with img_rows*img_cols input.
        Output: Keras Model.'''
   
    inputs = Input(shape=(img_rows, img_cols, 3))
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = Concatenate()([Convolution2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4])
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate()([Convolution2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3])
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate()([Convolution2D(64, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2])
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate()([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1])
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Convolution2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model
   
   
def image_prediction(unet, fpath, fname):

    ''' Split and predict input image.
        Input: U-net Model, file path, file name.
        Output: list of filenames:
        [0] - predicted image of the same size.
        [1] - predicted image of the same size with painted splitting lines.
        [2:]- predicted crops of U-net in-out size.'''
    
    # load img
    img = cv2.imread(str(Path(fpath,fname)))
    img_y, img_x, _ = img.shape

    # divide for crop_size squares, write to array
    min_size = min(img_x, img_y)

    if min_size<crop_size:
        zoom = crop_size/min_size
        new_x = int(img_x*zoom)
        new_y = int(img_y*zoom)
        img = cv2.resize(img,(new_x, new_y))
    img_y, img_x, _ = img.shape

    if img_x!=crop_size:
        n_crops_x = int(img_x/crop_size)
        dx = int((img_x - n_crops_x*crop_size)/n_crops_x) # смещение X для каждого кадра
    else:
        n_crops_x=0
        dx=0

    if img_y!=crop_size:
        n_crops_y = int(img_y/crop_size)
        dy = int((img_y - n_crops_y*crop_size)/n_crops_y) # смещение Y для каждого кадра
    else:
        n_crops_y=0
        dy=0
        
    res_img = np.zeros_like(img)

    xmin=0
    ymax=0
    img_n=0
    shapes = np.zeros(((n_crops_x+1)*(n_crops_y+1), 4),'int')

    for x in range(n_crops_x):
        xmax = xmin + crop_size
        ymin=0    
        for y in range(n_crops_y):
            ymax = ymin + crop_size
            shapes[img_n,:] = [xmin, xmax, ymin, ymax]
            img_n+=1
            ymin = ymax - dy
        ymin = img_y - crop_size
        ymax = img_y    
        shapes[img_n,:] = [xmin, xmax, ymin, ymax]
        img_n+=1
        xmin = xmax - dx
    xmin = img_x - crop_size
    xmax = img_x
    ymin=0    
    for y in range(n_crops_y):
        ymax = ymin + crop_size
        shapes[img_n,:] = [xmin, xmax, ymin, ymax]
        img_n+=1
        ymin = ymax - dy
    ymin = img_y - crop_size
    ymax = img_y  
    shapes[img_n,:] = [xmin, xmax, ymin, ymax]
    img_n+=1

    # predict
    result_fnames=[]
    result_fnames.append('pred_res_'+fname) # [0]
    result_fnames.append('pred_net_'+fname) # [1]
    
    # save predicted crops
    for rec in range(shapes.shape[0]):
            xmin, xmax, ymin, ymax = shapes[rec,:]
            X = img[ymin:ymax, xmin:xmax, :]
            X = np.expand_dims(X,0)        
            y_pred = unet.predict(X/255.)        
            cv2.imwrite(str(Path(fpath,'pred_'+str(rec)+'_'+fname)), (y_pred[0,]*255.).astype('uint8'))
            result_fnames.append('pred_'+str(rec)+'_'+fname) # [2:]
            res_img[ymin:ymax, xmin:xmax, :] = (y_pred[0,]*255.).astype('uint8')

    # plot lines
    for rec in range(shapes.shape[0]):
            xmin, xmax, ymin, ymax = shapes[rec,:]            
            img = cv2.line(img, (xmin,ymin), (xmin,ymax), (0,255,255), 3)
            img = cv2.line(img, (xmin,ymin), (xmax,ymin), (0,255,255), 3)
            img = cv2.line(img, (xmax,ymin), (xmax,ymax), (0,255,255), 3)
            img = cv2.line(img, (xmin,ymax), (xmax,ymax), (0,255,255), 3)
    cv2.imwrite(str(Path(fpath,'pred_net_'+fname)), img) 
    
    # save joined prediction
    cv2.imwrite(str(Path(fpath,'pred_res_'+fname)), res_img)    
    
    return result_fnames

    

def image_preprocess(bot, unet, save_path, save_name, chat_id=False):
    ''' Manage input image prediction process and operate TG bot if (chat_id != False).
        Output: None'''

    # message: user waits
    if chat_id: bot.send_message(chat_id, 'Картинка получена, ожидайте.')
    
    # predict image, get list of files
    result_fnames = image_prediction(unet, save_path, save_name)
    
    if len(result_fnames)>4:
    
        # several crops predicted            
        for file in result_fnames:
            capt=''                
            if file == result_fnames[1]:                
                capt = 'Это схема деления вашей картинки, дальше идут обработанные фрагменты - всего будет %s шт по 512х512 пикс.' % str(len(result_fnames)-2)
            if file == result_fnames[0]:                
                capt = 'Это обработанная картинка.'
                
            photo = open(Path(save_path, file), 'rb')
            if chat_id: bot.send_photo(chat_id, photo, caption = capt)
            del photo
            
    else:
    
        # only one crop predicted
        capt = 'Это обработанная картинка.'
        photo = open(Path(save_path, result_fnames[0]), 'rb')
        if chat_id: bot.send_photo(chat_id, photo, caption = capt)
        del photo
        
    result_fnames.append(save_name)

    return result_fnames
            
