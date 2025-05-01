# see https://python.langchain.com/docs/how_to/custom_embeddings/
from typing import List
from langchain_core.embeddings import Embeddings
import requests 

# from ragInstrumentation import measure_execution_time

DEBUG = False

class LocalEmbeddings(Embeddings):
    """ParrotLink embedding model integration.

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of ParrotLink model to use.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_parrot_link import LocalEmbeddings

            embed = LocalEmbeddings(
                model="...",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if token-level streaming isn't supported.
    Embed multiple text:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            embed.embed_documents(input_texts)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if native async isn't supported.
    Async:
        .. code-block:: python

            await embed.aembed_query(input_text)

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            # TODO: Example output.

    """

    def queryLocal(self, input: List[str]) -> List[List[float]]:
        """
        Query the local embedding service.

        This method is intended to interact with the local embedding
        service to retrieve or process embeddings. The exact implementation
        details should be defined based on the specific requirements of the
        local service.

        Returns:
            Any: The result of the query to the local embedding service.
        """
        hdrs = {"Content-Type": "application/json"}
        data = {"model": self.model, "input": input, "encoding_format": "float"}
        try:
            response = requests.post(self.url, headers=hdrs, json=data)
            response.raise_for_status()
            if response.status_code == 200:
                if DEBUG: print(response.json())
                vectors = [x["embedding"] for x in response.json()["data"]]
                if DEBUG: print("vectors",vectors)
                return vectors
            else:
                print("Encode error:",response.status_code) #,self.url,hdrs,data)
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error querying local embedding service: {e}")
            return None
        

    def __init__(self, model: str, url: str = "http://localhost:8085/v1/embeddings"):
        self.model = model
        self.url = url
        # test api with dummy data
        test_data = ["Hello", "world"]
        response = self.queryLocal(test_data)
        if response is None:
            raise ValueError("Failed to connect to the local embedding service.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.queryLocal(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.queryLocal([text])[0]

    # optional: add custom async implementations here
    # you can also delete these, and the base class will
    # use the default implementation, which calls the sync
    # version in an async executor:

    # async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #     """Asynchronous Embed search docs."""
    #     ...

    # async def aembed_query(self, text: str) -> List[float]:
    #     """Asynchronous Embed query text."""
    #     ...
