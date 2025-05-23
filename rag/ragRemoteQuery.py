"""
This script provides tools for querying a remote database using a Retrieval-Augmented Generation (RAG) system. 
It includes functions for initializing the system, checking the database, and querying a language model (LLM).
Modules:
    - json
    - os
    - sys
    - pandas as pd
    - argparse
    - ragTextUtils as textUtils
    - ragDeployUtils as deployUtils
    - measure_execution_time, log_query from ragInstrumentation
Functions:
    - initialize(): Initializes the configuration for the application, setting up the database client, preprocessor, embedder, and language model.
    - checkDb(): Checks if the specified database collection exists and is accessible.
    - queryLlm(context, query, history, size=200): Queries the language model (LLM) with the given context, query, and history.
    - initQuery(context, query, size=200): Initializes a query using the provided context and query parameters.
    - followQuery(query, history, size=200): Executes a follow-up query using the provided LLM configuration.
Main Execution:
    - Parses command-line arguments for items, language, collection, embProvider, and llmProvider.
    - Initializes the configuration and database.
    - Continuously prompts the user for queries, processes them, and prints the answers.
"""

"""
Example for local search with annService:
  python3 rag/ragRemoteQuery.py -d localsearch -s projects/ksk.db -c ksk_1024

start annService like so:
 ./annService 1024 9001 ../data/ksk_1024_title_de.vec ../data/ksk_1024_chunk_de.vec

more vectors may be loaded

local embedder can be used with param: -P localllama

start embedder service like so (for llama-cpp):
/opt/llama/bin/llama-server -m /opt/llama/models/bge-m3-Q4_K_M.gguf -c 0 -b 1000 -ub 1000  --embeddings --port 8085

on tux3 do 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/llama/cpu/lib
first and use /opt/llama/cpu/bin, or select oneapi/gpu version 

"""

import pandas as pd
import argparse 

import ragTextUtils as textUtils
import ragDeployUtils as deployUtils
from ragInstrumentation import measure_execution_time, log_query

import time


############################
# supervisord stuff
import os
import subprocess
import atexit
import signal
import sys

from urllib.parse import urlparse, urlunparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#SUPERVISOR_CONF = os.path.join(BASE_DIR, "..", "sv", 'supervisord.conf')
SEARCH_CONF = os.path.join(BASE_DIR, "..", "sv", 'search.conf')
EMBED_CONF = os.path.join(BASE_DIR, "..", "sv", 'embed.conf')
LLM_CONF = os.path.join(BASE_DIR, "..", "sv", 'llm.conf')
SUPERVISORCTL = 'supervisorctl'
SUPERVISORD = 'supervisord'

supervised = [] # list of running supervised processes
sv_options = {"search":SEARCH_CONF,"embed":EMBED_CONF,"llm":LLM_CONF}

def wait_for_service(name: str = None):
    """
    Wait for a specific service to be ready.
    If no name is provided, exception
    """
    if name == None:
        raise ValueError("Name missing")
    config = sv_options.get(name,None)
    if config == None:
        raise ValueError(f"Service {name} not found")
    
    while True:
        try:
            output = subprocess.check_output(
                [SUPERVISORCTL, '-c', config, 'status'],
                stderr=subprocess.STDOUT
            ).decode()
            lines = output.strip().splitlines()
            not_ready = [line for line in lines if 'RUNNING' not in line]
            if not not_ready:
                print(f"Service {name} is RUNNING.")
                break
            print(f"Waiting for service {name}:")
            print("\n".join(not_ready))
            time.sleep(2)
        except subprocess.CalledProcessError as e:
            print("Waiting for supervisor socket to become ready...")
            time.sleep(1)

def start_supervisord(name: str = None):
    if name == None:
        raise ValueError("Name missing")
    config = sv_options.get(name,None)
    if config == None:
        raise ValueError(f"Service {name} not found")

    print(f"Starting supervisord {name}, {config}...")
    subprocess.Popen([SUPERVISORD, '-c', config])

def shutdown_supervisord():
    for name in supervised:
        config = sv_options.get(name,None)
        if config == None:
            raise ValueError(f"Service {name} not found")

        print(f"Shutting down supervisord {name}, {config}...")
        try:
            subprocess.run([SUPERVISORCTL, '-c', config, 'shutdown'],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Error shutting down supervisor:", e)
        time.sleep(1)


############################


DEBUG = False


config = {
    # maybe basedir needed for source information
    "basedir":'../docs/karlsruhe/ksk_extracted',
    "lang": "de",
    "dbCollection": "ksk_de" , # dbCollection = "ksk_en"
    "dbItems" : 5,
    "dbClient" : None,
    "preprocessor" : None,
    "embedder" : None,
    "llm" : None,
    "embProvider":None,
    "llmProvider":None,
    "llmModel":None,
    "dbProvider":None,
    "search": []
}

def sigint_handler(signum, frame):
    print("\\nCaught Ctrl-C (SIGINT)")
    shutdown_supervisord()
    sys.exit(0)



def initialize():
    """
    Initializes the configuration for the application.

    This function sets up the following components:
    - Database client using `deployUtils.VectorDb()`.
    - Preprocessor for text processing using `textUtils.PreProcessor` with the specified language.
    - Embedder model using `deployUtils.Embedder()`.
    - Language model (LLM) using `deployUtils.Llm` with the specified language.

    The function also calls `checkDb()` to ensure the database is properly set up.
    """
    if config["dbProvider"] == "localsearch":
        config["dbSearch"] = {
            "title": deployUtils.VectorDb(provider=config["dbProvider"],collection=f'{config["dbCollection"]}_title'),
            #"summary": deployUtils.VectorDb(provider=config["dbProvider"],collection=f'{config["dbCollection"]}_summary'),
            "chunk": deployUtils.VectorDb(provider=config["dbProvider"],collection=f'{config["dbCollection"]}_chunk')
        }
        config["dbClient"] = config["dbSearch"]["title"]
        import ragSqlUtils as sq
        import private_remote as pr
        if config["dbSqlite"] != None:
            connection_string = f'sqlite:///{config["dbSqlite"]}'
        else:
            connection_string = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
        config["sql"] = {
            "sq":sq,
            "db":sq.DatabaseUtility(connection_string)
        }
    elif config["dbProvider"] == "pysearch":
        config["dbSearch"] = {
            "title": deployUtils.VectorDb(provider=config["dbProvider"],collection=f'{config["dbCollection"]}_title')
        }
        config["dbClient"] = config["dbSearch"]["title"]
        import ragSqlUtils as sq
        import private_remote as pr
        connection_string = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
        config["sql"] = {
            "sq":sq,
            "db":sq.DatabaseUtility(connection_string)
        }
    else:
        config["dbClient"] = deployUtils.VectorDb(provider=config["dbProvider"],collection=config["dbCollection"])
    checkDb()
    # text stuff
    config["preprocessor"] = textUtils.PreProcessor(config["lang"])
    # models
    config["embedder"] = deployUtils.Embedder(provider=config["embProvider"])
    # need to check embedder for zilliz, due to stored embeddings with all-minilm12-v2
    if config["dbProvider"] == "zilliz":
        config["embedder"].model = "sentence-transformers/all-MiniLM-L12-v2"
        if DEBUG: print("Embedder changed to MiniLM for Zilliz")
        
    # llm
    config["llm"] = deployUtils.Llm(lang=config["lang"],provider=config["llmProvider"],model=config["llmModel"])
    if config["llmUrl"] != None:
        config["llm"].url = config["llmUrl"]
    if config["llmPort"] != None:
        if config["llm"].url:
            parsed_url = urlparse(config["llm"].url)
            new_netloc = f"{parsed_url.hostname}:{config['llmPort']}"
            config["llm"].url = urlunparse(parsed_url._replace(netloc=new_netloc))

def checkDb():
    """
    Checks if the specified database collection exists and is accessible.

    This function attempts to describe the collection specified in the configuration.
    If the collection does not exist or an error occurs, it raises a ValueError.

    Raises:
        ValueError: If the collection does not exist or an error occurs during the check.

    Prints:
        Collection details if DEBUG is enabled.
        Error message if the collection check fails.
    """
    # check collection exists
    try:
        collection = config["dbClient"].describeCollection()
        if DEBUG: print(collection)
        if collection["code"] != 0:
            print(f"Error on {collection}: {collection['code']}")
            raise ValueError
        if DEBUG: print("Collection OK")
    except Exception as e:
        print("Collection failed",e)
        raise ValueError


@log_query
def queryLlm(context, query, history,size=200):
    def queryLlm(context, query, history, size=200):
        """
        Queries the language model (LLM) with the given context, query, and history.

        Args:
            context (str): The context to provide to the LLM.
            query (str): The query to ask the LLM.
            history (list): The history of previous interactions with the LLM.
            size (int, optional): The maximum number of tokens for the response. Defaults to 200.

        Returns:
            tuple: A tuple containing the answer from the LLM and the number of tokens used.
        """
    answer, tokens = config["llm"].queryWithContext(context, query, history,size)
    return answer, tokens

@log_query
def initQuery(context, query, size=200):
    """
    Initializes a query using the provided context and query parameters.

    Args:
        context (str): The context in which the query is being made.
        query (str): The query string to be processed.
        size (int, optional): The size parameter for the query. Defaults to 200.

    Returns:
        tuple: A tuple containing the answer, tokens, and messages from the query.
    """
    answer, tokens, msgs, think  = config["llm"].initChat(context, query, size)
    if config["think"] and think != None:
        print("Reasoning:",think)
    return answer, tokens, msgs

@log_query
def followQuery(query, history, size=200):
    """
    Executes a follow-up query using the provided LLM configuration.

    Args:
        query (str): The query string to be followed.
        history (list): The history of previous interactions or messages.
        size (int, optional): The size parameter for the LLM follow-up chat. Defaults to 200.

    Returns:
        tuple: A tuple containing the answer, tokens, and messages from the LLM follow-up chat.
    """
    answer, tokens, msgs, think  = config["llm"].followChat(query, history, size)
    if config["think"] and think != None:
        print("Reasoning:",think)
    return answer, tokens, msgs


def retrieve_context(query):
    """
    Retrieves the context for the given query based on the search results.

    Args:
        query (str): The query string to be processed.

    Returns:
        str: The context string for the query.
    """
    embedding = config["embedder"].encode(query)
    searchVector = embedding["data"][0]["embedding"]
    if DEBUG: print("search vector:",searchVector)
    if config["dbProvider"] == "zilliz":
        searchResult = config["dbClient"].searchItem(searchVector, limit=config["dbItems"], fields=["itemId","title","file","meta","text"])
        if DEBUG: print(searchResult)
        files = [f["file"] for f in searchResult["data"]]
        if DEBUG: print(files)
        results = [(f["itemId"], f["title"], f["text"]) for f in searchResult["data"] if f["distance"] >= config["threshold"]]
    elif config["dbProvider"] == "localsearch":
        tsearchResult = config["dbSearch"]["title"].searchItem(searchVector, limit=config["dbItems"]*2)
        csearchResult = config["dbSearch"]["chunk"].searchItem(searchVector, limit=config["dbItems"]*5)
        tsearchResult = tsearchResult["data"] if tsearchResult != None else []
        csearchResult = csearchResult["data"] if csearchResult != None else []
        if DEBUG: print(tsearchResult,csearchResult)
        # limit by threshold
        tsearchResult = [r for r in tsearchResult if r["similarity"] >= config["threshold"]]
        csearchResult = [r for r in csearchResult if r["similarity"] >= config["threshold"]]
        if DEBUG: print("Thresholded:",tsearchResult,csearchResult)
        # !! id in search results is vector INDEX !!
        # wrong. csearch returns chunk indices, tsearch returns title indices
        # replace vector indices with item ids
        titleItems = config["sql"]["db"].search(config["sql"]["sq"].Item, filters=[config["sql"]["sq"].Item.itemIdx.in_([idx["id"] for idx in tsearchResult])])
        # !important!
        # Ensure the order of titleItems matches the order of tsearchResult
        tsearchResult_ids = [idx["id"] for idx in tsearchResult]
        titleItems = sorted(titleItems, key=lambda item: tsearchResult_ids.index(item.itemIdx))
        if DEBUG: print(titleItems)
        #
        for i,item in enumerate(titleItems):
            tsearchResult[i]["id"] = item.id                
        # chunks
        chunks = config["sql"]["db"].search(config["sql"]["sq"].Chunk, filters=[config["sql"]["sq"].Chunk.chunkIdx.in_([idx["id"] for idx in csearchResult])])
        # !important!
        # Ensure the order of chunks matches the order of csearchResult
        csearchResult_ids = [idx["id"] for idx in csearchResult]
        chunks = sorted(chunks, key=lambda chunk: csearchResult_ids.index(chunk.chunkIdx))
        # collect chunk ids
        chunkIds = [c.id for c in chunks]
        chunkTexts = [(s.chunkId,s.itemId,s.content) for s in config["sql"]["db"].search(config["sql"]["sq"].Snippet, 
                        filters=[config["sql"]["sq"].Snippet.chunkId.in_(chunkIds),
                        config["sql"]["sq"].Snippet.lang == config["lang"],
                        config["sql"]["sq"].Snippet.type == "content"
                    ])]
        # !important!
        # Ensure the order of chunks matches the order of csearchResult
        chunkTexts = sorted(chunkTexts, key=lambda ct: chunkIds.index(ct[0]))
        if DEBUG: print("Chunks:",chunkIds, chunkTexts)

        # TODO: find chunk and title item id in separate lists and merge them
        searchResult = sorted(csearchResult + tsearchResult, key=lambda obj: obj["similarity"], reverse=True)
        # Remove duplicates by keeping only the first occurrence of each id
        unique_ids = set()
        filtered_search_result = []
        for item in searchResult:
            if item["id"] not in unique_ids:
                unique_ids.add(item["id"])
                filtered_search_result.append(item)
        searchResult = filtered_search_result
        # already threshold
        # searchResult = [r for r in searchResult if r["similarity"] >= config["threshold"]]
        if DEBUG: print(searchResult)
        print("Filtered search result:",searchResult)
        #print("Filtered search result:",searchResult)
        if len(searchResult) > 0:
            #indices = [f["id"] for f in searchResult["data"]]
            # vector indices have been converted to itemIds already
            # restrict here to number given in config
            itemIds = [f["id"] for f in searchResult][:config["dbItems"]]
            if DEBUG: print("ItemIds:",itemIds)
            items = config["sql"]["db"].search(config["sql"]["sq"].Item, filters=[config["sql"]["sq"].Item.id.in_(itemIds)])
            if DEBUG: print(len(items), " items:",items)
            # here files is also item names
            files = [i.name for i in items]
            if DEBUG: print("Files:",files)
            # search for title and fulltext in one go
            titles = config["sql"]["db"].search(config["sql"]["sq"].Snippet,
                filters=[
                    config["sql"]["sq"].Snippet.lang == config["lang"],
                    config["sql"]["sq"].Snippet.itemId.in_(itemIds),
                        config["sql"]["sq"].Snippet.type == "title",
                        config["sql"]["sq"].Snippet.chunkId == None
                ]
            )
            if DEBUG: print([t.id for t in titles])
            print([(t.id,t.itemId,t.content) for t in titles])
            fulltexts = config["sql"]["db"].search(config["sql"]["sq"].Snippet,
                filters=[
                    config["sql"]["sq"].Snippet.lang == config["lang"],
                    config["sql"]["sq"].Snippet.itemId.in_(itemIds),
                        config["sql"]["sq"].Snippet.type == "content",
                        config["sql"]["sq"].Snippet.chunkId == None
                ]
            )
            if DEBUG: print([(t.id,t.itemId) for t in fulltexts])
            summaries = config["sql"]["db"].search(config["sql"]["sq"].Snippet,
                filters=[
                    config["sql"]["sq"].Snippet.lang == config["lang"],
                    config["sql"]["sq"].Snippet.itemId.in_(itemIds),
                        config["sql"]["sq"].Snippet.type == "summary",
                        config["sql"]["sq"].Snippet.chunkId == None
                ]
            )
            if DEBUG: print([(t.id,t.itemId) for t in summaries])
            # itemids may be larger than items!
            if config["brief"]:
                results = [(files[i], titles[i].content, "" if len(summaries) < (i + 1) else summaries[i].content) for i in range(len(items))]
            else:
                results = [(files[i], titles[i].content, "" if len(fulltexts) < (i + 1) else fulltexts[i].content) for i in range(len(items))]
            if DEBUG: print("Results:",results)
        else:
            results = []
            
    elif config["dbProvider"] == "pysearch":
        searchResult = config["dbSearch"]["title"].searchItem(searchVector, limit=config["dbItems"]*2)
        searchResult = searchResult["data"] if searchResult != None else []
        if DEBUG: print(searchResult)
        # wrong. csearch returns chunk indices, tsearch returns title indices
        if DEBUG: print(searchResult)
        if len(searchResult) > 0:
            #indices = [f["id"] for f in searchResult["data"]]
            indices = [f["id"] for f in searchResult]
            if DEBUG: print("Indices:",indices)
            items = config["sql"]["db"].find_items(indices)
            if DEBUG: print(items)
            itemIds = [i[0] for i in items][:config["dbItems"]]
            # here files is also item names
            files = [i[1] for i in items][:config["dbItems"]]
            if DEBUG: print(itemIds)
            # search for title and fulltext in one go
            titles = config["sql"]["db"].search(config["sql"]["sq"].Snippet,
                filters=[
                    config["sql"]["sq"].Snippet.lang == config["lang"],
                    config["sql"]["sq"].Snippet.itemId.in_(itemIds),
                        config["sql"]["sq"].Snippet.type == "title",
                        config["sql"]["sq"].Snippet.chunkId == None
                ]
            )
            if DEBUG: print([t.id for t in titles])
            fulltexts = config["sql"]["db"].search(config["sql"]["sq"].Snippet,
                filters=[
                    config["sql"]["sq"].Snippet.lang == config["lang"],
                    config["sql"]["sq"].Snippet.itemId.in_(itemIds),
                        config["sql"]["sq"].Snippet.type == "content",
                        config["sql"]["sq"].Snippet.chunkId == None
                ]
            )
            if DEBUG: print([t.id for t in fulltexts])
            results = [(files[i], titles[i].content, fulltexts[i].content) for i in range(len(itemIds))]
        else:
            results = []
            
    else:
        raise ValueError("Unknown dbProvider")
    
    if len(results) == 0:
        print("No relevant documents found")
        return "",[]

    context = ""
    for r in results:
        context = "\n".join([f"{r[0].split('_chunk')[0]}:{r[1]}",r[2],context])
    return context, files

# Step 5: Run the RAG system
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--items', default = 5)      # option that takes a value
    parser.add_argument('-l', '--lang',default = "de")      # option that takes a value
    parser.add_argument('-d', '--dbProvider',default = "localsearch")      # option that takes a value
    parser.add_argument('-c', '--collection',default = "ksk")      # option that takes a value
    parser.add_argument('-P', '--embProvider',default = "deepinfra")      # option that takes a value
    parser.add_argument('-p', '--llmProvider',default = "deepinfra")      # option that takes a value
    parser.add_argument('-m', '--llmModel',default = None)      # option that takes a value
    parser.add_argument('-r', '--llmPort',default = None, help="Optional LLM port number")      # option that takes a value
    parser.add_argument('-s', '--sqlite',default = None)      # option that takes a value
    parser.add_argument('-S', '--stream',action='store_true', help='Enable streaming')
    parser.add_argument('-t', '--think',action='store_true', help='Enable think output')
    parser.add_argument('-b', '--brief', action='store_true', help='Use summaries instead of fulltext')
    parser.add_argument('-T', '--threshold', default = .55, help='Search threshold')
    parser.add_argument('-u', '--llmUrl',default = None, help="LLM url")      # option that takes a value
    
    args = parser.parse_args()
    print(args.items, args.lang, args.collection) 

    config["lang"] = args.lang
    config["dbCollection"] = f"{args.collection}"
    config["dbItems"] = int(args.items)
    config["embProvider"] = args.embProvider
    config["llmProvider"] = args.llmProvider
    config["llmModel"] = args.llmModel
    config["dbProvider"] = args.dbProvider
    config["dbSqlite"] = args.sqlite
    config["stream"] = args.stream
    config["think"] = args.think
    config["brief"] = args.brief
    config["threshold"] = args.threshold
    config["llmUrl"] = args.llmUrl
    config["llmPort"] = args.llmPort
    if DEBUG: print(config)
    
    signal.signal(signal.SIGINT, sigint_handler)
    atexit.register(shutdown_supervisord)

    # start services, if required
    if config["dbProvider"] == "localsearch":
        supervised.append("search")
        if args.collection == None:
            raise ValueError("Collection name missing")
        if os.path.sep in args.collection:
            collection = args.collection
            config["dbCollection"] = os.path.basename(args.collection)
        else:
            collection = "/opt/llama/data/vectors/" + args.collection
        collection += f"_*{args.lang}.vec"
        #os.environ['RAG_SEARCH_ARGS'] = ' '.join(["1024","9001","/opt/llama/data/vectors/ksk_1024_*de.vec"])
        os.environ['RAG_SEARCH_ARGS'] = ' '.join(["1024","9001",collection])
        print("collection:",collection,config["dbCollection"])
        print(os.environ['RAG_SEARCH_ARGS'])
        start_supervisord("search")
        wait_for_service("search")
    
    if config["embProvider"] == "localllama":
        supervised.append("embed")
        start_supervisord("embed")
        wait_for_service("embed")

    if config["llmProvider"] == "localllama":
        supervised.append("llm")
        if args.llmModel == None:
            args.llmModel = "granite-3.3-2b-instruct" # default
        model_files = [f for f in os.listdir('/opt/llama/models') if f.startswith(args.llmModel)]
        if not model_files:
            raise FileNotFoundError(f"No model file starting with {config['llmModel']} found in /opt/llama/models")
        model_file = os.path.join('/opt/llama/models', model_files[0])
        print("model:",model_file)
        os.environ['RAG_LLM_MODEL'] = model_file # '/opt/llama/models/granite-3.3-2b-instruct-Q4_K_M.gguf'
        if args.llmPort != None:
            os.environ['RAG_LLM_PORT'] = str(args.llmPort)
            print("llmPort:",args.llmPort)
        start_supervisord("llm")
        wait_for_service("llm")


    #####################    
    initialize()

    msgHistory = []
    query = input("\nEnter your query: ")
    followUp = False
    while len(query) > 0:
        if not followUp:
            context, files = retrieve_context(query)
            if DEBUG: print(context)
            if context == "":
                print("No relevant documents found")
                query = input("\nEnter your query: ")
                continue
            answer, tokens, msgs = initQuery(context, query)
            followUp = True
            queryChain = query # init for updates
        else:
            # add assistant answer to msgs
            msgs.append({"role":"assistant","content":answer})
            queryChain = " ".join([queryChain, query])
            if query.startswith("U:"):
                query = query[2:]
                new_context, files = retrieve_context(queryChain)
                if DEBUG: print(new_context)
                print("Context has been updated with new data.")
                if new_context == "":
                    print("No relevant documents found")
                else:
                    msgs.append({"role":"user","content":f"The context has been updated with the following information\n\n{new_context}"})

            answer, tokens, msgs = followQuery(query,msgs)
            
        if answer == None:
            print("No answer found")    
        print("Answer:", answer,tokens, files)
        if DEBUG: print("History","no result" if msgs == None else msgs)
        query = input("\nEnter your query (With optional N: or U:): ")
        if query.startswith("N:"):
            query = query[2:]
            followUp = False
            msgHistory = []

    
    
# text = db.search(sq.Snippet,filters=[sq.Snippet.itemId==19,sq.Snippet.type=='content',sq.Snippet.chunkId==None])
# tx = text[0].content
# x = llm.summarizeJson(tx)
# xj = json.loads(x[0][7:][:-4])
# xj["facts"]
# xj["summary"]
