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
    return context


def getPrompt(query, results):
    context = create_context(results)
    full_prompt = f"""
        You are a helpful AI assistant.

        Answer the user's question using the provided context.

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
        """
    return full_prompt