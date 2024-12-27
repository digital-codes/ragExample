import os
import sys
import requests
from ragInstrumentation import measure_execution_time

import private_remote as pr

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
            self.model = pr.deepInfra["embMdl"]
            self.url = pr.deepInfra["embUrl"]
            self.engine = None
        elif provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = 'sentence-transformers/all-MiniLM-L12-v2'
                self.engine = SentenceTransformer(self.model, device='cpu')
                self. url = None
                # embeddings = model.encode(sentences)  # Returns a NumPy array
            except ModuleNotFoundError:
                raise ValueError("Error: Please install sentence_transformers")
        else:
            raise ValueError("Invalid provider")

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
            data = {"model": self.model, "input": input, "encoding_format": "float"}
            response = requests.post(self.url, headers=hdrs, json=data)
            if response.status_code == 200:
                return response.json()
            else:
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
    def __init__(self, provider: str = "deepinfra", lang="de"):
        if provider == "deepinfra":
            self.api_key = pr.deepInfra["apiKey"]
            self.model = pr.deepInfra["lngMdl"]
            self.url = pr.deepInfra["lngUrl"]
        elif provider == "openai":
            self.api_key = pr.openAi["apiKey"]
            self.model = pr.openAi["lngMdl"]
            self.url = pr.openAi["lngUrl"]
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

    @measure_execution_time
    def initChat(self, context, query, size=100):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        msgs = []
        msgs.append({
            "role": "system", 
            "content": """
                You are an expert assistant designed to provide detailed and accurate responses 
                based on user queries and retrieved context. 
                If the retrieved context is insufficient or ambiguous, 
                ask for clarification or provide a logical extrapolation."
            """
                })
        msgs.append({
            "role": "system", 
            "content": f"""
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
            self.lang = "german" if lang == "de" else "english"
            self.collection = f'{collection}_{lang}'
            self.url = f'https://{pr.zilliz["cluster"]}.serverless.{pr.zilliz["region"]}.cloud.zilliz.com'
        else:
            raise ValueError("Invalid provider")

        print(self.url)

    @measure_execution_time
    def describeCollection(self, collection):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {"collectionName": collection}
        url = f"{self.url}/v2/vectordb/collections/describe"
        response = requests.post(url, headers=hdrs, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code)
            return None

    @measure_execution_time
    def indexCollection(self, collection, field):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {
            "collectionName": collection,
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

    @measure_execution_time
    def indexDescribeCollection(self, collection, field):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {"collectionName": collection, "indexName": field}
        url = f"{self.url}/v2/vectordb/indexes/describe"
        response = requests.post(url, headers=hdrs, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code)
            return None

    @measure_execution_time
    def indexListCollection(self, collection):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {"collectionName": collection}
        url = f"{self.url}/v2/vectordb/indexes/list"
        response = requests.post(url, headers=hdrs, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code)
            return None

    @measure_execution_time
    def statCollection(self, collection):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {"collectionName": collection}
        url = f"{self.url}/v2/vectordb/collections/get_stats"
        response = requests.post(url, headers=hdrs, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code)
            return None

    @measure_execution_time
    def upsertItem(self, collection, item):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {"collectionName": collection, "data": [item]}
        url = f"{self.url}/v2/vectordb/entities/upsert"
        response = requests.post(url, headers=hdrs, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code)
            return None

    @measure_execution_time
    def searchItem(self, collection, vectors, limit=3, fields=["*"], groupByFile=True):
        """_summary_

        Args:
            collection (_type_): _description_
            vectors (_type_): _description_
            limit (int, optional): _description_. Defaults to 3.
            fields (list, optional): _description_. Defaults to ["*"].
            groupByFile (bool, optional): _description_. Defaults to True.
            L2      Smaller L2 distances indicate higher similarity.
            IP      Larger IP distances indicate higher similarity.
            COSINE  Larger cosine value indicates higher similarity.
        """

        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {
            "collectionName": collection,
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

    @measure_execution_time
    def queryText(self, collection, condition, limit=3, fields=["*"]):
        hdrs = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if collection == None:
            collection = self.collection
        data = {
            "collectionName": collection,
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
