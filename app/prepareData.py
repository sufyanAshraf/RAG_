
def create_content(place):
    content = (
        f"name: {place['name']}, "
        f"city: {place['city']} "
        f"region: {place['region']} "
        f"rating: {str(place["rating"])}"
        f"distance: {place["distance"]}"
        f"Services: {', '.join(place['services'])}. "
        f"description: {place['description']}"
    )

    return content

def create_filter(place, content, count, catagory):
    distance = float(place["distance"].replace("km", "").strip())
    
    record = {
        "_id": str(count),
        "chunk_text": content,       # must match field_map's "text" -> "chunk_text"
        "category": catagory.lower(),
        "name": (place["name"]).lower(),
        "city": (place["city"]).lower(),
        "region": (place["region"]).lower(),
        "services": [service.lower() for service in place.get("services", [])],  # list 
        "rating": float(place["rating"]),
        "distance": distance, 
    }

    return record

def data_prepration_for_embadddings(data, categories = ["spa", "hotel", "restaurant"]):
    """
    Create embeddings for a list of texts using the specified model.

    Args:
        texts (list): A list of strings to create embeddings for.
        model (str): The name of the embedding model to use."""
 
    records = []
    cat = categories
    count = 1

    for j in range(len(data)):
        for i, place in enumerate(data[j]):

            content = create_content(place)

            catagory = cat[j] 

            record = create_filter(place, content, count, catagory)

            records.append(record)
            count += 1

    return records