from .appInitlize import* 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG API", version="0.1.0") 
db, model = initlize_db_and_llm()
history_manager = HistoryManager(model, window_size=6)  # session wiring comes later
processor = QueryProcessor()

 
@app.post("/", response_model=chatResponse)
async def chat(request: chatRequest) -> chatResponse:   
    query = request.query
    retrieval_query = build_retrieval_query(query, history_manager)
    result = processor.process(model, retrieval_query)

    if not result["allowed"]:
        return chatResponse(response=result["message"])

    filter = result["filter"]


    # result, msg = guardrails_query(model , query) 
    # if result:
    #     return chatResponse(response=msg)

    # retrieval_query = build_retrieval_query(query, history_manager)
    
    # # create filter
    # filter_obj = queryCreator()
    # filter = filter_obj.create_query(model, retrieval_query)

    try:
        results = db.hybrid_search(retrieval_query, filter)
    except Exception as e:
        logger.error("Error: retriving query from Pineonce")
        raise RuntimeError(f"Pinecone API failed: {e}")
 
    logger.info("Successfull query database") 

    if not results:
        return chatResponse(response="Sorry we are unable to find anything related to your query")
        
    # create prompt
    full_prompt = getPrompt(query, results, history_manager.get_context())

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

    history_manager.add_turn(ConversationTurn(query=query, response=response))
 
    return chatResponse(response=response) 


@app.get("/health")
def health_check() -> dict[str, str]: 
    """Health check endpoint to verify that the API is running.
        returns
            A dictionary indicating the health status of the API.
    """
    logger.info("Health check requested")
    return {"status": "ok"}
 
from langchain_groq import ChatGroq
from .eval import RAGEvaluator
from .evalData import eval_queries
@app.get("/eval")
def quality_test(): 
    groq_api_key, pinecone_api_key, key = read_api_key_from_config()
    
    LLM_model =  ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-safeguard-20b")
     
    evaluator = RAGEvaluator(eval_queries=eval_queries, 
        model=LLM_model,
        key = key,
        db = db,
        model_class = model,
        processor = processor,
        dataset_name="RAG_Eval_Dataset2",
    )
    results = evaluator.evaluate(
        experiment_prefix="RAG_Eval_Langsmith_Native1",
        max_concurrency=2,
    )
    print(results)
    return {"data": "ok"}
 