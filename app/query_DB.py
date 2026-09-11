

def pc_search(query, index, namespace, filter):

    results = index.search(
        namespace=namespace,
        query={
            "top_k": 5,
            "inputs": {"text": query},
            "filter":  filter
        }
    )

    return results

