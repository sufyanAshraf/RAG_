from pinecone import Pinecone, ServerlessSpec 

from .logger import logger


class dataBase:
    # _instance = None

    # def __new__(cls):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(self, key, index_name = "rag-hotel", namespace = "services-providers" ):
        # if not hasattr(self, "index"):
        #     self.index = None

        if not key:
            raise ValueError("API key is required for Pinecone.")
        self.key = key

        self.index_name = index_name
        self.namespace = namespace

        try:
            self.pc = Pinecone(api_key=self.key) #connection
        except Exception as e:
            logger.error(f"Error initializing pinecone: {e}")
            raise Exception(f"Failed to initialize pinecone: {e}")
        
        logger.info("Pinecone Successfully connected")

        self.index_flag = self.create_index()

        try:
            self.index = self.get_index()
        except Exception as e:
            logger.error(f"Error pinecone index: {e}")
            raise Exception(f"Failed to get pinecone index: {e}")

    def get_index_flag(self):
        return self.index_flag
        

    def create_index(self):
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

    def get_index(self):
        index = self.pc.Index(self.index_name)
        return index

    def initial_upsert(self, records):
        # Pinecone recommends batches of ~96 records or fewer for upsert_records
        batch_size = 90 
    
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            self.index.upsert_records(records=batch, namespace=self.namespace)
    

    def pc_search(self, query, filter):

        results = self.index.search(
            namespace=self.namespace,
            query={
                "top_k": 5,
                "inputs": {"text": query},
                "filter":  filter
            }
        )

        return results