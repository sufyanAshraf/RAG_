from .logger import logger

def format_history(history):
    return "\n".join(
        f"User: {turn.query}\nAssistant: {turn.response}"
        for turn in history
    )

def create_context(results):
    hits = results["result"]["hits"]

    context = ""

    for hit in hits:
        fields = hit.get("fields", {})

        context += f"""
            Name: {fields.get('name', '')}
            Category: {fields.get('Category', '')}
            City: {fields.get('city', '')}
            Region: {fields.get('region', '')}
            Rating: {fields.get('rating', '')}
            Distance: {fields.get('distance', '')}
            Services: {fields.get('services', '')}
            Description: {fields.get('description', '')}

            ---
            """
    # logger.info("context:" + context)  
    return context


def getPrompt(query, results, history=None):
    context = create_context(results)
    if not context:
        logger.error("Error in context")
        return None

    conversation = format_history(history or [])
    conversation_section = conversation or "No previous conversation."

    full_prompt = f"""
        You are a helpful AI assistant.

        Answer the user's question using the provided context.

        Previous conversation:
        {conversation_section}

        Context:
        {context}

        Question:
        {query}

        Answer:


        Do not:
        - Wrap the response in single or double quotes.
        - Return escaped characters such as \n or \u202f.
        - Include unnecessary introductory or closing remarks.
        - Use Unicode spaces.

        Use normal spaces and newlines. 
        use bullet points
        """
    return full_prompt