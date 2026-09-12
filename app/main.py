from .appInitlize import*


app = FastAPI(title="RAG API", version="0.1.0") 


@app.post("/", response_model=chatResponse)
async def chat(request: chatRequest) -> chatResponse:
    groq_api_key, pinecone_api_key = read_api_key_from_config()

    model = GroqModel(groq_api_key)   

    # Store the vectors in the database
    
    namespace = "services-providers" 
    db = dataBase(pinecone_api_key)
    index_flag = db.create_index()
    index = db.get_index()
    
    if index_flag == False:
        read_data = readData()
        data = read_data.readjson()
    
        # Create embeddings for the data
        index = create_embeddings(data, index, namespace )

    # create filter
    query = request.query
    filter_obj = queryCreator()
    filter = filter_obj.create_query(model, query)

    try:
        results = db.pc_search(query, index, namespace, filter)
    except Exception as e:
        logger.error("Error: retriving query from Pineonce")
        raise RuntimeError(f"Pinecone API failed: {e}")
 
    logger.info("Successfull query database") 

    # create prompt
    full_prompt = getPrompt(query, results)

    if not full_prompt:
        logger.error("Error in context")
        raise ValueError("Prompt Value cannot be empty") 

    # -------------------------
    # 8. Call Groq
    # -------------------------
    try:
        response = model.invoke_model(
            full_prompt=full_prompt
        ) 
        logger.info("Successfull response")
    except Exception as e:
        logger.error(f"Groq API failed: {e}")
        raise RuntimeError(f"Groq API failed: {e}")
 
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
 