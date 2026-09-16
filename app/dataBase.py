from pinecone import Pinecone, ServerlessSpec, SchemaBuilder

from .logger import logger


class dataBase:
    # _instance = None

    # def __new__(cls):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(self, key, index_name = "rag-hotel", namespace = "services-providers" ):
        
        if not key:
            raise ValueError("API key is required for Pinecone.")
        self.key = key

        self.index_name = index_name
        self.fts_index_name = f"{index_name}-fts"
        self.namespace = namespace

        try:
            self.pc = Pinecone(api_key=self.key) #connection
        except Exception as e:
            logger.error(f"Error initializing pinecone: {e}")
            raise Exception(f"Failed to initialize pinecone: {e}")
        
        logger.info("Pinecone Successfully connected")

        self.index_flag = self.create_index()
        self.fts_index_flag = self.create_fts_index()

        try:
            self.index = self.get_index()
        except Exception as e:
            logger.error(f"Error pinecone index: {e}")
            raise Exception(f"Failed to get pinecone index: {e}")

        try:
            self.fts_index = self.get_fts_index()
        except Exception as e:
            logger.error(f"Error pinecone fts index: {e}")
            raise Exception(f"Failed to get pinecone fts index: {e}")
        

    def get_index_flag(self):
        return self.index_flag
    
    def get_fts_index_flag(self):
        return self.fts_index_flag
        

    def create_index(self):
        """ 
        """
        index_flag = False
        try:
            if not self.pc.has_index(self.index_name):
                self.pc.create_index_for_model(
                    name=self.index_name,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model":"llama-text-embed-v2",
                        "field_map":{"text": "chunk_text"}
                    }
                )
                logger.info("Index created")
            else:
                logger.info("Index already exist")
                index_flag = True
        except Exception as e:
            logger.error(f"Error pinecone index creation failed: {e}")
            raise 

        return index_flag

    def create_fts_index(self):
        """
        BM25 / keyword-search index (separate from the dense index).
        Uses the newer schema-based Documents API (pc.indexes.create),
        NOT the legacy pc.create_index(). Requires pinecone-client v10+.
        """
        index_flag = False
        try:
            if not self.pc.has_index(self.fts_index_name):
                schema = (
                    SchemaBuilder()
                    .add_string_field(name="chunk_text", full_text_search={"language": "en"})
                    .build()
                )
                self.pc.indexes.create(
                    name=self.fts_index_name,
                    schema=schema,
                    deployment={
                        "deployment_type": "managed",
                        "cloud": "aws",
                        "region": "us-east-1"
                    },
                    read_capacity={"mode": "OnDemand"}
                )
                logger.info("FTS (BM25) index created")
            else:
                logger.info("FTS index already exist")
                index_flag = True
        except Exception as e:
            logger.error(f"Error pinecone fts index creation failed: {e}")
            raise
 
        return index_flag
    

    def get_index(self):
        index = self.pc.Index(self.index_name)
        return index

    def get_fts_index(self):
        index = self.pc.Index(self.fts_index_name)
        return index

    def initial_upsert(self, records):
        # Pinecone recommends batches of ~96 records or fewer for upsert_records
        batch_size = 90 
    
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            self.index.upsert_records(records=batch, namespace=self.namespace) 
            self.fts_index.documents.upsert(namespace=self.namespace, documents=batch)
            

    def dense_search(self, query, filter, top_k):
        """ search for  
        """
        results = self.index.search(
            namespace=self.namespace,
            query={
                "top_k": top_k,
                "inputs": {"text": query},
                "filter":  filter
            }
        )

        return results

    def bm25_search(self, query, filter, top_k):
 
        results = self.fts_index.documents.search(
            namespace=self.namespace,
            top_k=top_k,
            score_by=[{"type": "text", "fields": ["chunk_text"], "query": query}],
            filter=filter,
            include_fields=["chunk_text"]  # required or only _id/_score come back
        )
 
        return results

    def hybrid_search(self, query, filter, top_k=5, rerank_top_n=5):
        """
        Runs dense (pc_search) + BM25 (bm25_search), merges the candidates
        by _id, and reranks them into one relevance-ordered list.
 
        NOTE: the two responses have different shapes —
        - dense (vectors/records API): dense_results["result"]["hits"], each
          hit is a dict with "_id" and "fields"
        - bm25 (documents API):        bm25_results.matches, each match is an
          object with ._id and attributes for each included field
        """
        dense_results = self.dense_search(query, filter, top_k)
        bm25_results = self.bm25_search(query, filter, top_k)
 
        # merge + dedupe by _id, normalizing both shapes into one dict form
        candidates = {}
 
        for hit in dense_results["result"]["hits"]:
            try:
                hit_id = hit["id"]
            except KeyError:
                hit_id = hit["_id"]
            candidates[hit_id] = {
                "id": hit_id,
                "text": hit["fields"]["chunk_text"]
            }
 
        for match in bm25_results.matches:
            candidates[match._id] = {
                "id": match._id,
                "text": getattr(match, "chunk_text", "")
            }
 
        merged = list(candidates.values())
 
        if not merged:
            return []
 
        try:
            reranked = self.pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=merged,
                rank_fields=["text"],
                top_n=rerank_top_n,
                return_documents=True
            )
        except Exception as e:
            logger.error(f"Error reranking hybrid search results: {e}")
            raise
 
        hits = []
        for item in reranked.data:
            hits.append({
                "_id": item.document["id"],
                "_score": item.score,
                "fields": {"chunk_text": item.document["text"]}
            })
 
        return {"result": {"hits": hits}}
 