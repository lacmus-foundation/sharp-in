from typing import List
from functools import lru_cache
from pydantic import BaseSettings
import os

# https://github.com/tiangolo/fastapi/issues/508#issuecomment-532360198

class Config(BaseSettings):
    project_name:   str = "Sharp-In"
    api_prefix:     str = "/api/v0"
    version:        str = "0.1.0"
    debug:          bool = True
    debug_chat:     int = os.environ['TG_CHAT_SHARP_IN']
    weights:        str = "./weights/sharp_in_weights.h5"
    token:          str = os.environ['TG_TOKEN_SHARP_IN']    
    save_path:      str = "./tmp"
    crop_size:      int = 512
    
    #-------------------------------------------------
    WEBHOOK_HOST:   str = '09b7ad867bb0.ngrok.io' # w/o "https://"
    #-------------------------------------------------

@lru_cache()
def get_config() -> Config:
    return Config()