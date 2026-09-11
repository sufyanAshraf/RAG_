
def create_embeddings(data, index, namespace ):
    """
    Create embeddings for a list of texts using the specified model.

    Args:
        texts (list): A list of strings to create embeddings for.
        model (str): The name of the embedding model to use."""
 
    records = []
    cat = ["spa", "hotel", "restaurant"]
    count = 1

    for j in range(len(data)):
        for i, place in enumerate(data[j]):

            content = (
                f"Name: {place['name']}, "
                f"City: {place['city']} "
                f"Region: {place['region']} "
                f"rating: {str(place["rating"])}"
                f"distance: {place["distance"]}"
                f"Services: {', '.join(place['services'])}. "
                f"Description: {place['description']}"
            )

            record = {
                "_id": str(count),
                "chunk_text": content,       # must match field_map's "text" -> "chunk_text"
                "Category": cat[j],
                "name": place["name"],
                "city": place["city"],
                "region": place["region"],
                "services": place["services"],  # list 
                "rating": str(place["rating"]),
                "distance": place["distance"],
                "description": place["description"],
            }

            records.append(record)
            count += 1

    # Pinecone recommends batches of ~96 records or fewer for upsert_records
    batch_size = 90 

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        index.upsert_records(records=batch, namespace=namespace)

    return index