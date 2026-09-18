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