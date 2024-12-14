# ragExample
Example for retrieval augmented genereration (RAG)

## Tools

RAG fundamentals: combine LLM with knowledge base

 * Searchable knowledge base: Text search, **vector database**, knowledge graph
 * LLM: locally installed e.g. via Ollama, **OpenAI**

### Search engines

Tested [chromadb](https://docs.trychroma.com/) and [elasticsearch](https://www.elastic.co/)

Install elasticsearch from [here](https://www.elastic.co/downloads/elasticsearch)

Install chromadb as documented [here](https://docs.trychroma.com/getting-started)

Elasticsearch provided better results, might be related to embedding algorithm 
(default on chromadb, transformer on elastic. chroma could use same embedder as well ...)

Elasticsearch needs more than 8GB of memory (dies on cloud VM). 
Chromadb works well on same machine (in principle)

#### Embeddings

Embedding options for search engines to be investigated. So far, built-in embedder from chromadb vs 
[sentence-transformer](https://huggingface.co/sentence-transformers) with 
[deepset/gbert-base](https://huggingface.co/deepset/gbert-base) for German. 

**Use something else for non-German application like [all-MiniLM-L6-v1](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1)**



### LLMs

Tested locally [Ollama](https://ollama.com/) with [llama3.2, 3B](https://ollama.com/library/llama3.2) 
Works but very slow without GPU

Tested with OpenAI API. Works. *Inital testing, maybe 50 trial queries ~ 50k tokens, 0.30€*




