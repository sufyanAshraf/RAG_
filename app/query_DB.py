

def pc_search(query, index, namespace):

    results = index.search(
        namespace=namespace,
        query={
            "top_k": 5,
            "inputs": {"text": query},
            "filter": {
                "Category": {"$eq": "spa"},
                "city": {"$eq": "Helsinki"}
            }
        }
    )

    return results

