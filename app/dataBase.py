from pinecone import Pinecone, ServerlessSpec 

from .logger import logger


class dataBase:
    # _instance = None

    # def __new__(cls):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(self, key):
        # if not hasattr(self, "index"):
        #     self.index = None

        if not key:
            raise ValueError("API key is required for Pinecone.")
        self.key = key

        try:
            self.pc = Pinecone(api_key=self.key)
        except Exception as e:
            logger.error(f"Error initializing pinecone: {e}")
            raise Exception(f"Failed to initialize pinecone: {e}")
        
        logger.info("Pinecone Successfully initilize")

    def create_index(self, index_name):
        index_flag = False
        try:
        
            if not self.pc.has_index(index_name):
                self.pc.create_index_for_model(
                    name=index_name,
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
            logger.info("Index fail")
            raise 

        return self.pc, index_flag