import os
import sys
import requests
import time
from ragInstrumentation import measure_execution_time

import private_remote as pr
import ragConfig as cfg

DEBUG = False


class Embedder:
    """A class to handle text embedding using either a remote or local provider.

    Attributes:
    -----------
    api_key : str
        The API key for the remote provider (only used if provider is "deepinfra").
    model : str
        The model name or path used for embedding.
    url : str or None
        The URL for the remote provider's embedding service (only used if provider is "deepinfra").
    engine : SentenceTransformer or None
        The local embedding engine (only used if provider is "local").

    Methods:
    --------
    __init__(provider: str = "deepinfra"):
        Initializes the Embedder with the specified provider.
    
    encode(input: str) -> dict:
        Encodes the input text and returns the embedding as a dictionary.
    
    supports remote or local provider. local vs deepinfra on same model all-MiniLM-L12-v2 
    provide comparable results. 400 vectors from ksk_full collection differ only after 6th decimal digit.
    """
    @measure_execution_time
    def __init__(self, provider: str = "deepinfra"):
        if provider == "deepinfra":
            self.api_key = pr.deepInfra["apiKey"]
            self.model = cfg.deepInfra["embMdl"]
            self.url = cfg.deepInfra["embUrl"]
            self.size = cfg.deepInfra["embSize"]
            self.toks = cfg.deepInfra["embToks"]
            self.engine = None
        elif provider == "huggingface":
            self.api_key = pr.huggingface["apiKey"]
            self.model = cfg.huggingface["embMdl"]
            self.url = cfg.huggingface["embUrl"]
            self.size = cfg.huggingface["embSize"]
            self.toks = cfg.huggingface["embToks"]
            self.engine = None
            print("Warning: Huggingface embeddings are non-standard.")
        elif provider == "openai":
            self.api_key = pr.openAi["apiKey"]
            self.model = cfg.openAi["embMdl"]
            self.url = cfg.openAi["embUrl"]
            self.size = cfg.openAi["embSize"]
            self.toks = cfg.openAi["embToks"]
            self.engine = None
            print("Warning: OpenAI embeddings are non-standard.")
        elif provider == "localllama":
            self.api_key = pr.localllama["apiKey"]
            self.model = cfg.localllama["embMdl"]
            self.url = cfg.localllama["embUrl"]
            self.size = cfg.localllama["embSize"]
            self.toks = cfg.localllama["embToks"]
        elif provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = 'sentence-transformers/all-MiniLM-L12-v2'
                self.engine = SentenceTransformer(self.model, device='cpu')
                self.size = 384
                self.toks = 512
                self. url = None
                # embeddings = model.encode(sentences)  # Returns a NumPy array
            except ModuleNotFoundError:
                raise ValueError("Error: Please install sentence_transformers")
        else:
            raise ValueError("Invalid provider")
        self.provider = provider
        self.size = self.get_size() 

    def get_size(self):
        """
        Returns:
            int: The size of the embedding vectors.
        """
        return self.size
        if "all-minilm-l12" in self.model.lower():
            return 384
        elif "jina-embeddings-v2-base" in self.model.lower():
            return 768
        elif "bge-m3" in self.model.lower():
            return 1024
        else:
            raise ValueError("Unknown model") 

    @measure_execution_time
    def encode(self, input):
        """
        Encodes the given input into a specified format.
        like: {"data":["embedding":vector]}

        If the URL is not provided and the engine is available, it uses the engine to encode the input.
        Otherwise, it sends a POST request to the specified URL with the input data.

        Args:
            input (str): The input data to be encoded.

        Returns:
            dict or None: A dictionary containing the encoded data if successful, otherwise None.

        Raises:
            requests.exceptions.RequestException: If the request to the URL fails.
        """
        if (self.url == None) and (self.engine != None):
            response = {"data": [{"embedding": list(self.engine.encode([input])[0].astype(float))}]}
            return response
        else:
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            if self.provider == "huggingface":
                data = {"inputs": input}
            else:           
                data = {"model": self.model, "input": input, "encoding_format": "float"}
            if self.provider == "openai":
                # default dims is 1536 ...
                data["dimensions"] = 384
            # get response
            response = requests.post(self.url, headers=hdrs, json=data)
            if response.status_code == 200:
                if DEBUG: print(response.json())
                if self.provider == "huggingface":
                    # plain vector
                    embedding = response.json()
                    return {"data":[{"embedding":embedding}]}
                else:  
                    return response.json()
            else:
                print("Encode error:",response.status_code)
                return None


class Llm:
    """
    A class to interact with different language model providers for various NLP tasks.
    Attributes:
        api_key (str): The API key for the selected provider.
        model (str): The model identifier for the selected provider.
        url (str): The API endpoint URL for the selected provider.
        lang (str): The language for responses ("german" or "english").
        provider (str): The name of the provider ("deepinfra" or "openai").
        temperature (float): The temperature setting for the model's responses.
    Methods:
        getModel():
            Returns the current model identifier.
        setModel(model):
            Sets a new model identifier.
        query(query, size=100):
            Sends a query to the language model and returns the response and token usage.
        summarize(text, size=500):
            Summarizes the given text and returns the summary and token usage.
        translate(text, src="english"):
            Translates the given text from the source language to the target language and returns the translation and token usage.
        queryWithContext(context, query, msgHistory=[], size=100):
            Sends a query with context to the language model and returns the response, token usage, and updated message history.
        initChat(context, query, size=100):
            Initializes a chat session with context and returns the response, token usage, and initial message history.
        followChat(query, msgHistory=[], size=100):
            Continues a chat session with a follow-up query and returns the response, token usage, and updated message history.
    """
    def __init__(self, provider: str = "deepinfra", lang="de", model=None):
        if provider == "deepinfra":
            self.api_key = pr.deepInfra["apiKey"]
            self.model = cfg.deepInfra["lngMdl"]
            self.url = cfg.deepInfra["lngUrl"]
        elif provider == "openai":
            self.api_key = pr.openAi["apiKey"]
            self.model = cfg.openAi["lngMdl"]
            self.url = cfg.openAi["lngUrl"]
        elif provider == "localllama":
            self.api_key = pr.localllama["apiKey"]
            self.model = cfg.localllama["lngMdl"]
            self.url = cfg.localllama["lngUrl"]
        elif provider == "huggingface":
            if model == None:
                raise ValueError("Huggingface model not provided")
            model_index = cfg.huggingface["lngMdl"].index(model)
            if model_index == -1:
                raise ValueError("Invalid model")
            self.api_key = pr.huggingface["apiKey"]
            # dedicated endpoints use different model name
            if cfg.huggingface["endpoint"][model_index] == "dedicated":
                self.model = "tgi"        
            else:
                self.model = cfg.huggingface["lngMdl"][model_index]
            self.url = cfg.huggingface["lngUrl"][model_index]
            # need to check endpoint availability
            testUrl = self.url.split("/v1")[0]
            while True:
                rsp = requests.get(testUrl,headers={"Authorization":f"Bearer {self.api_key}"})
                # returns 401 if wrong api key or 503 if not ready
                if rsp.status_code == 200: 
                    if DEBUG: print("Huggingface Ready")
                    break
                elif rsp.status_code == 401:
                    raise ValueError("Huggingface Unauthorized")
                elif rsp.status_code == 503:
                    print("Huggingface Not ready",rsp.status_code)
                    time.sleep(5)
                else:
                    raise ValueError("Unknown Huggingface error:", rsp.status_code)
        else:
            raise ValueError("Invalid provider")
        self.lang = "german" if lang == "de" else "english"
        self.provider = provider
        self.temperature = 0.2

    def getModel(self):
        """
        Retrieve the model instance.

        Returns:
            object: The model instance.
        """
        return self.model
    
    def setModel(self, model):
        """
        Sets the model for the instance.

        Args:
            model: The model to be set.
        """
        self.model = model

    @measure_execution_time
    def query(self, query, size = 100):
        """
        Executes a query to an intelligent assistant model and returns the response.

        Args:
            query (str): The query string to be sent to the model.
            size (int, optional): The maximum number of words in the response. Defaults to 100.

        Returns:
            tuple: A tuple containing the response text (str) and the total number of tokens used (int),
                   or None if the request was unsuccessful.
        """
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        richQuery = f"""
        You are an intelligent assistant.
        The question is:
        {query}
        Respond in {self.lang} language with a limit of {size} {self.lang} words.
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": richQuery, "temperature": self.temperature}],
        }
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens
        else:
            return None

    @measure_execution_time
    def summarize(self, text, size=500):
        """
    Summarizes the given text using an external API.

        Args:
            text (str): The text to be summarized.
            size (int, optional): The maximum number of words for the summary. Defaults to 500.

        Returns:
            tuple: A tuple containing the summarized text (str) and the total number of tokens used (int) if the request is successful.
            None: If the request fails.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.

        Decorators:
            measure_execution_time: Measures the execution time of the function.

        Notes:
            - The function sends a POST request to an external API with the provided text and other parameters.
            - The API key and other configurations are expected to be set as instance variables.
            - The function prints the query if DEBUG mode is enabled.
            
        """
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        richQuery = f"""
        You are an intelligent assistant.
        Summarize the following {self.lang} text into {self.lang}:
        {text}
        Respond in {self.lang} language. Limit the summary to {size} {self.lang} words.
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": richQuery, "temperature": self.temperature}],
        }
        if DEBUG:
            print(richQuery)
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens
        else:
            return None

    @measure_execution_time
    def summarizeJson(self, text, size=500):
        """
    Summarizes the given text using an external API.

        Args:
            text (str): The text to be summarized.
            size (int, optional): The maximum number of words for the summary. Defaults to 500.

        Returns:
            tuple: A tuple containing the summarized text (str) and the total number of tokens used (int) if the request is successful.
            None: If the request fails.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.

        Decorators:
            measure_execution_time: Measures the execution time of the function.

        Notes:
            - The function sends a POST request to an external API with the provided text and other parameters.
            - The API key and other configurations are expected to be set as instance variables.
            - The function prints the query if DEBUG mode is enabled.
            
        """
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        richQuery = f"""
        You are an intelligent assistant.
        Extract the key facts from the following {self.lang} text for the following 
        categories: impact, cost, funding, human resources, dates.
        Then summarize the following {self.lang} text into {self.lang}:
        {text}
        Respond in json with summary text in {self.lang} language as key "summary". Limit the summary to {size} {self.lang} words.
        Put the list of extracted facts in the key "facts" as key-value pairs "category":"exctracted fact" Use {self.lang} for fact values.        
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": richQuery, "temperature": self.temperature}],
        }
        if DEBUG:
            print(richQuery)
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens
        else:
            return None

    @measure_execution_time
    def getFacts(self, text):
        """
    Extract facts from the given text using an external API.

        Args:
            text (str): The text to be summarized.

        Returns:
            tuple: A tuple containing the summarized text (str) and the total number of tokens used (int) if the request is successful.
            None: If the request fails.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.

        Decorators:
            measure_execution_time: Measures the execution time of the function.

        Notes:
            - The function sends a POST request to an external API with the provided text and other parameters.
            - The API key and other configurations are expected to be set as instance variables.
            - The function prints the query if DEBUG mode is enabled.
            
        """
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        richQuery = f"""
        You are an intelligent assistant.
        Extract the key facts from the following {self.lang} text for the following 
        categories: impact, cost, funding, human resources, dates.
        {text}
        Respond in json with a list of facts as key-value pairs "category":"exctracted fact".
        Use {self.lang} for fact values.
        Use english for category names: impact, cost, funding, hr, dates. Omit categories with no facts.
        Do not add comments to the json response.
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": richQuery, "temperature": self.temperature}],
        }
        if DEBUG:
            print(richQuery)
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens
        else:
            return None

    @measure_execution_time
    def preview(self, text, size=50):
        """
    Makes a very compact preview of the given text using an external API.

        Args:
            text (str): The text to be summarized.
            size (int, optional): The maximum number of words for the summary. Defaults to 50.

        Returns:
            tuple: A tuple containing the summarized text (str) and the total number of tokens used (int) if the request is successful.
            None: If the request fails.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.

        Decorators:
            measure_execution_time: Measures the execution time of the function.

        Notes:
            - The function sends a POST request to an external API with the provided text and other parameters.
            - The API key and other configurations are expected to be set as instance variables.
            - The function prints the query if DEBUG mode is enabled.
            
        """
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        richQuery = f"""
        You are an intelligent assistant.
        Make a very compact news-feed of the following {self.lang} text into {self.lang}:
        {text}
        Respond in {self.lang} language. Limit the repsonse to {size} {self.lang} words.
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": richQuery, "temperature": self.temperature}],
        }
        if DEBUG:
            print(richQuery)
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens
        else:
            return None

    @measure_execution_time
    def translate(self, text, src="english"):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        richQuery = f"""
        You are an intelligent assistant.
        Translate the following {src} text into {self.lang}:
        {text}
        Respond in {self.lang} language, translation only, no comments. Do not add any leading or trailing comments on translation process.
        """
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": richQuery, "temperature": self.temperature}],
        }
        if DEBUG:
            print(richQuery)
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens
        else:
            return None
    
    @measure_execution_time
    def queryWithContext(self, context, query, msgHistory=[], size=100):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        richQuery = f"""
        You are an intelligent assistant. Use the following {self.lang} context to answer the question in {self.lang}.
        {context}
        The {self.lang} question is:
        {query}
        Respond in {self.lang} language. Limit your response to {size} {self.lang} words.
        """
        msg = {"role": "user", "content": richQuery}
        msgHistory.append(msg)
        data = {
            "model": self.model,
            "messages": [msg],  # msgHistory,
            "temperature": self.temperature,
        }
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            if DEBUG:
                print(data)
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            msgHistory.append(
                {"role": data["choices"][0]["message"]["role"], "content": text}
            )
            return text, tokens
        else:
            print("LLM failed:",response.status_code)
            return None, None

    # Note: huggings face does not allow multiple system roles. 
    # First one is optional, then alternate between user and assistant

    @measure_execution_time
    def initChat(self, context, query, size=100):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        msgs = []
        msgs.append({
            "role": "system", 
            "content": """You are an expert assistant designed to provide detailed and accurate responses 
                based on user queries and retrieved context. 
                If the retrieved context is insufficient or ambiguous, 
                ask for clarification or provide a logical extrapolation."""
                + f"""
                Context in {self.lang}:\n
                {context}
                """
                })
        msgs.append({
            "role": "user", 
            "content": f"""
                The {self.lang} question is:
                {query}
                Respond in {self.lang} language. Limit your response to {size} {self.lang} words.
                Offer assistance for follow-up question.
                """
                })
                
        data = {
            "model": self.model,
            "messages": msgs,  # msgHistory,
            "temperature": self.temperature,
        }
        if self.provider == "huggingface":
            data["stream"] = False
        if DEBUG:
            print(data)
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            if DEBUG:
                print(data)
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens, msgs
        else:
            print("LLM failed:",response.status_code)
            return None, None, None

    @measure_execution_time
    def followChat(self, query, msgHistory=[], size=100):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        msgHistory.append({
            "role": "user", 
            "content": f"""
                The {self.lang} question is:\n
                {query}
                
                Respond in {self.lang} language. Limit your response to {size} {self.lang} words. 
                Silently verify the question is related to the context provided. 
                Do not offer assistance if the question is unrelated.
                """
            })
        data = {
            "model": self.model,
            "messages": msgHistory,  # msgHistory,
            "temperature": self.temperature,
        }
        if self.provider == "huggingface":
            data["stream"] = False
        response = requests.post(self.url, headers=hdrs, json=data)
        if response.status_code == 200:
            data = response.json()
            if DEBUG:
                print(data)
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["total_tokens"]
            return text, tokens, msgHistory
        else:
            print("LLM failed:",response.status_code)
            return None, None, None



# https://cloud.zilliz.com/orgs/org-vuubdaymoyjvtgcqjczdsp/projects/proj-11d29d1ea430702a07c431/clusters/in03-eb450554ac4fcc5/collections/ksk/playground?collection=ksk&type=QUERY_DATA
class VectorDb:
    """ make sure to check parameter names. rest api is different from python"""
    def __init__(self, provider: str = "zilliz", lang = "de", collection="ksk"):
        if provider == "zilliz":
            self.api_key = pr.zilliz["apiKey"]
            self.collection = f'{collection}_{lang}'
            self.url = f'https://{cfg.zilliz["cluster"]}.serverless.{cfg.zilliz["region"]}.cloud.zilliz.com'
        elif provider == "localsearch":
            self.url = cfg.localsearch["url"]
            response = requests.get(self.url)
            if response.status_code == 200:
                collections = response.json()
                # localize collection
                locCollection = f'{collection}_{lang}'
                if locCollection in collections:
                    self.collection = collections.index(locCollection)
                else:
                    raise ValueError(f"Invalid collection:{locCollection}")
            else:
                raise ValueError("Service error:",response.status_code)
        elif provider == "pysearch":
            self.url = cfg.pysearch["url"]
            # support only 1 collection
            self.collection = 0
        else:
            raise ValueError("Invalid provider")
        self.provider = provider
        self.lang = lang

    @measure_execution_time
    def describeCollection(self):
        if self.provider == "zilliz":
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {"collectionName": self.collection}
            url = f"{self.url}/v2/vectordb/collections/describe"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(response.status_code)
                return None
        elif self.provider == "localsearch":
            self.url = cfg.localsearch["url"]
            response = requests.get(self.url)
            if response.status_code == 200:
                return {"code": 0}
            else:
                raise ValueError("Service error:",response.status_code)
        elif self.provider == "pysearch":
            self.url = cfg.pysearch["url"]
            response = requests.get(self.url)
            if response.status_code == 200:
                return {"code": 0}
            else:
                raise ValueError("Service error:",response.status_code)
        else:
            raise ValueError("Invalid provider")

    @measure_execution_time
    def indexCollection(self, field):
        if self.provider == "zilliz":
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {
                "collectionName": self.collection,
                "indexParams": [
                    {
                        "metricType": "L2",
                        "fieldName": field,
                        "indexName": field,
                        "indexConfig": {"index_type": "AUTOINDEX"},
                    }
                ],
            }
            url = f"{self.url}/v2/vectordb/indexes/create"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(response.status_code)
                return None
        else:
            raise ValueError("Invalid provider")

    @measure_execution_time
    def indexDescribeCollection(self, field):
        if self.provider == "zilliz":
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {"collectionName": self.collection, "indexName": field}
            url = f"{self.url}/v2/vectordb/indexes/describe"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(response.status_code)
                return None
        else:
            raise ValueError("Invalid provider")

    @measure_execution_time
    def indexListCollection(self):
        if self.provider == "zilliz":
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {"collectionName": self.collection}
            url = f"{self.url}/v2/vectordb/indexes/list"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(response.status_code)
                return None
        else:
            raise ValueError("Invalid provider")

    @measure_execution_time
    def statCollection(self):
        """
        Collects statistics for a given collection based on the provider.

        This method collects statistics for a specified collection from either the "zilliz" or "localsearch" provider.
        If the provider is "zilliz", it sends a POST request to the specified URL with the collection name and returns the response JSON.
        If the provider is "localsearch", it sends a GET request to the specified URL and checks if the collection exists in the response.

        Args:
            collection (str): The name of the collection to collect statistics for. If None, the default collection is used.

        Returns:
            dict or int or None: 
                - If the provider is "zilliz" and the request is successful, returns the response JSON as a dictionary.
                - If the provider is "localsearch" and the request is successful, returns the index of the collection in the response list, or -1 if the collection is not found.
                - If the request fails, returns None.

        Raises:
            ValueError: If the provider is not "zilliz" or "localsearch".
        """
        if self.provider == "zilliz":
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {"collectionName": self.collection}
            url = f"{self.url}/v2/vectordb/collections/get_stats"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(response.status_code)
                return None
        else:
            raise ValueError("Invalid provider")

    @measure_execution_time
    def upsertItem(self, item):
        if self.provider == "zilliz":
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {"collectionName": self.collection, "data": [item]}
            url = f"{self.url}/v2/vectordb/entities/upsert"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(response.status_code)
                return None
        else:
            raise ValueError("Invalid provider")

    @measure_execution_time
    def searchItem(self, vectors, limit=3, fields=["*"], groupByFile=True):
        """_summary_

        Args:
            vectors (_type_): _description_
            limit (int, optional): _description_. Defaults to 3.
            fields (list, optional): _description_. Defaults to ["*"].
            groupByFile (bool, optional): _description_. Defaults to True.
            L2      Smaller L2 distances indicate higher similarity.
            IP      Larger IP distances indicate higher similarity.
            COSINE  Larger cosine value indicates higher similarity.
        """
        if DEBUG:
            print(len(vectors))
            if len(vectors) == 1:
                print(len(vectors[0]))
                
        if self.provider == "zilliz":

            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {
                "collectionName": self.collection,
                "data": [vectors],
                "limit": limit,
                "outputFields": fields,
            }
            if groupByFile:
                data["groupingField"] = "itemCode"
            url = f"{self.url}/v2/vectordb/entities/search"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                if response.json()["code"] == 0:
                    return response.json()
                raise ValueError(response.json())
            else:
                print(response.status_code)
                return None
        elif self.provider == "localsearch":
            data = {
                "collection": self.collection,
                "vectors": vectors,
                "limit": limit
            }
            response = requests.post(self.url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                if DEBUG: print(response.status_code)
                return None
        elif self.provider == "pysearch":
            data = {
                "collection": self.collection,
                "vectors": vectors,
                "limit": limit
            }
            response = requests.post(self.url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                if DEBUG: print(response.status_code)
                return None
        else:
            raise ValueError("Invalid provider")

    @measure_execution_time
    def queryText(self, condition, limit=3, fields=["*"]):
        if self.provider == "zilliz":
            hdrs = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {
                "collectionName": self.collection,
                "filter": condition,
                "limit": limit,
                "outputFields": fields,
            }
            url = f"{self.url}/v2/vectordb/entities/query"
            response = requests.post(url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(response.status_code)
                return None
        else:
            raise ValueError("Invalid provider")
