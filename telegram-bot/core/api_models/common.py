from pydantic import BaseModel
#from fastapi_utils.enums import StrEnum
from typing import List
from enum import auto

class Pong(BaseModel):
    pong: str = "Sharp_In web API, version 0.1.0"