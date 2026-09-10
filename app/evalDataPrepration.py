# import numpy as np
# from .evalData import eval_queries
# from ragas import EvaluationDataset
# from .logger import logger


# def indexing(embedding_records):
#     ids = [r["id"] for r in embedding_records]
#     id_to_record = {r["id"]: r for r in embedding_records}
#     return ids, id_to_record
    

# def retrieve(query,  model, index, ids, id_to_record, k=5):
#     query_vector = np.array(model.embed_query(query)).astype("float32").reshape(1, -1)
#     distances, indices = index.search(query_vector, k)
#     retrieved_ids = [ids[i] for i in indices[0]]
#     contexts = [id_to_record[rid]["metadata"]["content"] for rid in retrieved_ids]
#     return contexts

# def generate(query, contexts, gen_llm):
#     prompt = (
#         "Answer the question using ONLY the context below. "
#         "If the answer isn't in the context, say you don't know.\n\n"
#         f"Context:\n{chr(10).join(contexts)}\n\nQuestion: {query}"
#     )
#     response = gen_llm.invoke(prompt) 
     
#     return response.content

 
# def create_evaluation_dataset(model, index, gen_llm, embedding_records, k=5): 
#     ids, id_to_record = indexing(embedding_records)  
 
#     dataset = []

#     for item in eval_queries:
#         query = item["query"]
#         contexts = retrieve(query, model=model, index=index, ids=ids, id_to_record=id_to_record, k=k)
#         response = generate(query, contexts,gen_llm)

#         dataset.append({
#             "user_input": query,
#             "retrieved_contexts": contexts,
#             "response": response,
#             "reference": item["reference"],
#         })

#     evaluation_dataset =  EvaluationDataset.from_list(dataset)
#     return evaluation_dataset