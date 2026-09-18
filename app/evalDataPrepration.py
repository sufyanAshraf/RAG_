# import numpy as np
# from .evalData import eval_queries
# from ragas import EvaluationDataset
# from .logger import logger


# def indexing(embedding_records):
#     ids = [r["id"] for r in embedding_records]
#     id_to_record = {r["id"]: r for r in embedding_records}
#     return ids, id_to_record
    

# def retrieve1(query, db, model, k=5): 
#     # result = processor.process(model, query)
#     # filter = result["filter"]
#     # results = db.hybrid_search(query, filter)
#     # full_prompt = getPrompt(query, results, "No previous conversation.")
#     # response = model.invoke_model(
#     #             full_prompt=full_prompt
#     #         ) 
#     # return response, results
    
#     results = db.search_db(query)
#     contexts = []
#     if "result" in results and "hits" in results["result"]:
#         for hit in results["result"]["hits"]:
#             contexts.append(hit["fields"]["chunk_text"])
#     return contexts

# def generate(query, contexts,model):
#     prompt = (
#         "Answer the question using ONLY the context below. "
#         "If the answer isn't in the context, say you don't know.\n\n"
#         f"Context:\n{chr(10).join(contexts)}\n\nQuestion: {query}"
#     )
#     response = model.invoke(prompt)
#     return response.content
 
from .prompt import getPrompt 

def create_context( results):
    
    hits = results["result"]["hits"]

    contexts = []

    for hit in hits:
        fields = hit.get("fields", {})

        chunk_text = fields.get("chunk_text", "")

        if chunk_text:
            contexts.append(chunk_text)

    return contexts

def retrieve(query, db, model, processor): 
    result = processor.process(model, query)

    filter = result["filter"]
    results = db.hybrid_search(query, filter)

    full_prompt = getPrompt(query, results, "No previous conversation.")

    response = model.invoke_model(full_prompt=full_prompt) 

    return response, create_context(results) 