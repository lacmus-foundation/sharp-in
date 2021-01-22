from fastapi import APIRouter, HTTPException, UploadFile, File  
from fastapi.responses import FileResponse
from core.unet.u_net import *
from core.config import get_config
import telebot
import shutil
import os
from pathlib import Path
from pydantic import BaseModel

# instantiate model
crop_size = get_config().crop_size
unet = define_unet(crop_size, crop_size, optimizer=Adam(learning_rate=1e-4))
unet.load_weights(get_config().weights)

# instantiate TG bot
bot = telebot.TeleBot(get_config().token)

# set TG chat for tests
if get_config().debug:
    debug_chat_id = get_config().debug_chat
else:
    debug_chat_id = None

router = APIRouter()

class Message(BaseModel):
        message: dict        

async def delete_tmp_files(targ_folder):
    '''Delete all files in a predictions folder'''
    for file in os.listdir(targ_folder): 
        os.remove(Path(targ_folder, file))    


@router.post("/infer")
async def predict_on_image(image: UploadFile = File(...)):

    # tested with:
    #  curl -F "image=@C:\IMAG5746.jpg" http://f963acd213e2.ngrok.io/api/v0/infer --output 121212.jpg

    # uploaded file path-name    
    save_name = image.filename    
    save_path = get_config().save_path
    
    # clean up old prediction files
    try:
        await delete_tmp_files(save_path)        
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    # save uploaded file   
    with open(Path(save_path, save_name), "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # call predicting routine
    try:        
        results = image_preprocess(bot, unet, save_path, save_name, debug_chat_id)          
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    
    return FileResponse(str(Path(save_path, results[0])))


@router.post("/tg")  
async def tg_predict(msg: Message):

    # get chat_id and qty of images from TG
    msg = msg.dict()['message']
    chat_id = msg['chat']['id']       
    
    # process only if image exists
    if "photo" in msg.keys(): 

        # set variables                    
        n_photos = len(msg['photo'])-1    
        save_name = msg['photo'][n_photos]['file_id'] + ".jpg"
        save_path = get_config().save_path
        
        # get image from TG server
        file_info = bot.get_file(msg['photo'][n_photos]['file_id'])
        downloaded_file = bot.download_file(file_info.file_path)        
        
        # save image locally
        with open(Path(save_path, save_name),'wb') as new_file:
            new_file.write(downloaded_file) 
        
        # call predicting routines
        try:
            results = image_preprocess(bot, unet, save_path, save_name, chat_id)        
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))
        
        # clean up all prediction files
        try:
            await delete_tmp_files(save_path)        
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))
            
        return {"detail":"file processed"}

    else:
        bot.send_message(chat_id, 'Пришлите мне картинку, пожалуйста, я в другом виде данные не понимаю.')
        