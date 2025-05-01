import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
import private_remote as pr 

LOCAL_EMBDDINGS = True

if not LOCAL_EMBDDINGS:
    from langchain_community.embeddings import DeepInfraEmbeddings
    embMdl = "BAAI/bge-m3"
    embeddings = DeepInfraEmbeddings(model_id=embMdl,deepinfra_api_token=pr.deepInfra["apiKey"])
else:
    import localEmbeddings as LE
    embeddings = LE.LocalEmbeddings("bge-m3-Q4_K_M")


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

storeName = "faiss_test_index"

from uuid import uuid4
from langchain_core.documents import Document


def initStore(name):
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this mo\
    rning.",
        metadata={"source": "tweet"},
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high \
    of 62 degrees.",
        metadata={"source": "news"},
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applica\
    tions!",
        metadata={"source": "tweet"},
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.\
    ",
        metadata={"source": "news"},
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    vector_store.save_local("faiss_test_index")

    return vector_store    



if os.path.exists(storeName):
    print(f"Loading {storeName}...")       
    vector_store = FAISS.load_local(
        storeName, embeddings, allow_dangerous_deserialization=True
    )
else:
    vector_store = initStore(storeName)

# search 

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

results = vector_store.similarity_search_with_relevance_scores(
    "LangGraph is the best framework for building stateful, agentic apps",
    k=3,
    filter={"source": "tweet"},
)
print("################################################")
# 
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
    print(f"  [ID={res.id}]")
    
    # Find related document
    document = vector_store.docstore.search(res.id)
    if document:
        print(f"Document found: {document.page_content} [{document.metadata}]")
    else:
        print(f"No document found with ID: {res.id}")

