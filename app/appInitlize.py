from fastapi import FastAPI
from pydantic import BaseModel 
import configparser 
from .logger import logger  
from .models import GroqModel   
from .readData import readData
from .embeddingsCreator import create_embeddings
from .dataBase import dataBase 
from .prompt import getPrompt
from .query_creator import queryCreator

def read_api_key_from_config() -> str:
    """Read the API key from the config.ini file."""
    logger.info("Reading API key from config file")
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config.get("KEYS", "groq_api_key") , config.get("KEYS", "pinecone_api_key")

class chatRequest(BaseModel):
    query: str

class chatResponse(BaseModel):
    response: str