from fastapi import FastAPI
from pydantic import BaseModel 
import configparser 
from .logger import logger  
from .models import GroqModel   
from .readData import readData
from .prepareData import data_prepration_for_embadddings
from .dataBase import dataBase 
from .prompt import getPrompt
# from .query_creator import queryCreator
from .historyManager import HistoryManager
# from .guardrails import QueryGuardrail
from .queryProcessor import QueryProcessor

# guardrail = QueryGuardrail()

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

class ConversationTurn(BaseModel):
    query: str
    response: str

def initlize_db_and_llm():

    groq_api_key, pinecone_api_key = read_api_key_from_config()
    
    model = GroqModel(groq_api_key)   
    
    # Store the vectors in the database    
    db = dataBase(pinecone_api_key)
    index_flag = db.get_index_flag() 
    
    if index_flag == False:
        read_data = readData()
        data = read_data.readjson()
    
        # Create embeddings for the data
        prep_data = data_prepration_for_embadddings(data)
        db.initial_upsert(prep_data)

    return db, model

def build_retrieval_query(query, history_manager: HistoryManager):
    return history_manager.get_retrieval_query(query)

# def guardrails_query(model , query): 

#     guard_result = guardrail.check(model, query)
#     if not guard_result["allowed"]:
#         logger.info(
#             f"Query blocked by guardrail: category={guard_result['category']} "
#             f"reason={guard_result['reason']}"
#         )
#         return True , guard_result["message"]

#     return False, None