from fastapi import FastAPI
from pydantic import BaseModel 
import configparser 
from .logger import logger  
from .models import GroqModel   
from .readData import readData
from .embeddingsCreator import create_embeddings
from .dataBase import dataBase
from .query_DB import pc_search 
from .prompt import getPrompt
from .query_creator import queryCreator

def read_api_key_from_config() -> str:
    """Read the API key from the config.ini file."""
    logger.info("Reading API key from config file")
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config.get("KEYS", "groq_api_key") , config.get("KEYS", "pinecone_api_key")

app = FastAPI(title="RAG API", version="0.1.0") 

class chatRequest(BaseModel):
    query: str

class chatResponse(BaseModel):
    response: str


@app.post("/", response_model=chatResponse)
async def chat(request: chatRequest) -> chatResponse:
    groq_api_key, pinecone_api_key = read_api_key_from_config()

    model = GroqModel(groq_api_key)   

    # Store the vectors in the database
    index_name = "rag-hotel"
    namespace = "hotels" 
    db = dataBase(pinecone_api_key)
    pc, index_flag = db.create_index(index_name)
    index = pc.Index(index_name)

    if index_flag == False:
        read_data = readData()
        data = read_data.readjson()
    
        # Create embeddings for the data
        index = create_embeddings(data, index, namespace )

    # query model
    query = request.query
    filter_obj = queryCreator()
    filter = filter_obj.create_query(model, query)

    try:
        results = pc_search(query, index, namespace, filter)
    except:
        logger.info("Error: retriving query from Pineonce")
        raise
 
    logger.info("Successfull query database")  
    # create prompt
    full_prompt = getPrompt(query, results)
    if not full_prompt:
        logger.error("Error in context")
        raise

    # -------------------------
    # 8. Call Groq
    # -------------------------
        
    response = model.invoke_model(
        full_prompt=full_prompt
    ) 
    logger.info("Successfull response")
     
 
    return chatResponse(response=response) 


@app.get("/health")
def health_check() -> dict[str, str]: 
    """Health check endpoint to verify that the API is running.
        returns
            A dictionary indicating the health status of the API.
    """
    logger.info("Health check requested")
    return {"status": "ok"}
 
 
@app.get("/eval")
def quality_test(): 
    # groq_api_key, huggingface_api_key = read_api_key_from_config()
    
    # model =  ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-safeguard-20b")
    # # docs = read_documents_from_file()
    # embeddings_model = EmbeddingsModel(local_model=True)
    # embedder = embeddings_model.get_model()  
    
    # read_data = readData()
    # data = read_data.readjson()

    # # Create embeddings for the data
    # embedding_records, vectors = create_embeddings(data, embedder)

    # # Store the vectors in the database
    # db = dataBase()
    # index = db.store_vectors(vectors)
    
    # # run evals
    # evaluation_dataset = create_evaluation_dataset(embedder, index, model, embedding_records)
    # result = evaluateWithRagas(evaluation_dataset, model, embedder)
     
    # logger.info("successfull")
    # df = result.to_pandas()

    # # data = df.to_dict(orient="records") 
    
    # logger.info(df)

    return {"data": "ok"}
 