from groq import Groq
from langchain_groq import ChatGroq
from .logger import logger  
 
class GroqModel: 
    def __init__(self, key=None, model_name="openai/gpt-oss-safeguard-20b", temperature=0.0):
        """Initializes the GroqModel with the provided API key and model name.  
        Args:
            key (str): The API key for Groq.
            model_name (str): The name of the model to use. Defaults to "llama-3.1-8b-instant".
        """
        if not key:
            raise ValueError("API key is required for GroqModel.")
        self.key = key
        self.model_name = model_name
        self.llm = self.get_model()
    
    def get_model(self): 
        """Returns a ChatGroq instance with the specified API key and model name.
        Returns:
            ChatGroq: An instance of the ChatGroq class configured with the API key and model name.
        """
        if not self.key:
            raise ValueError("API key is required to initialize the model.")

        try:
            return ChatGroq(groq_api_key=self.key, model_name=self.model_name)
        except Exception as e:
            logger.error(f"Error initializing ChatGroq: {e}")
            raise Exception(f"Failed to initialize ChatGroq: {e}")

    
    def get_model_name(self):
        """Returns the name of the model being used.
        Returns:
            str: The name of the model.
        """
        return self.model_name

 
    def invoke_model(self, full_prompt, system_prompt = None):
        """
        Invokes the model with the given messages.
        
        Args:
            messages (list): A list of messages to send to the model.
        
        Returns:
            str: The response from the model.
        """ 

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        if not full_prompt:
            logger.error(f"Error occurred Full prompt is required : {e}")
            raise ValueError("Full prompt is required to invoke the model.")
         
        try:
            response = self.llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt}])
            return response.content
        
        except Exception as e:
            logger.error(f"Error occurred while invoking the model: {e}")
            raise Exception(f"Failed to invoke the model: {e}")

