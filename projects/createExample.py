import os
"""
This script creates an example project using the provided configuration file (example.json).
It performs the following tasks:
1. Reads the configuration file.
2. Initializes necessary utilities and modules.
3. Creates a project directory and database.
4. Inserts project details into the database.
5. Processes items and their associated tags, snippets, and chunks.
6. Generates vectors for titles, chunks, and summaries.
Modules:
- os: Provides a way of using operating system dependent functionality.
- sys: Provides access to some variables used or maintained by the interpreter.
- json: Provides methods to work with JSON data.
- numpy: Provides support for large, multi-dimensional arrays and matrices.
- ragSqlUtils: Custom module for database utilities.
- ragDeployUtils: Custom module for deployment utilities.
- ragTextUtils: Custom module for text processing utilities.
Configuration:
- exampleConfigFile: Path to the configuration file (example.json).
Classes and Functions:
- PreProcessor: Class for text preprocessing.
- Embedder: Class for embedding text using a specified provider.
- DatabaseUtility: Class for database operations.
- Project: Class representing a project.
- Tag: Class representing a tag.
- Item: Class representing an item.
- Snippet: Class representing a snippet of text.
- Chunk: Class representing a chunk of text.
Workflow:
1. Read the configuration file.
2. Initialize the text preprocessor and embedder.
3. Create the project directory if it doesn't exist.
4. Delete any existing database file and create a new one.
5. Insert project details into the database.
6. Process each item in the configuration:
    - Create and insert tags.
    - Create and insert items.
    - Create and insert snippets (title, facts, summary, content).
    - Create and insert chunks of text.
7. Generate vectors for titles, chunks, and summaries.
"""
import sys
import json
import numpy as np

# add path to rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag")))
import ragSqlUtils as sq
import ragDeployUtils as rag
import ragTextUtils as rt

exampleConfigFile = "example.json"

try:
    with open(exampleConfigFile) as f:
        exampleConfig = json.load(f)
except Exception as e:
    print(f"Error reading {exampleConfigFile}: {e}")
    sys.exit()

print("Creating example project:", exampleConfig["name"])

# text stuff
preprocessor = rt.PreProcessor()

# get embedder. use different provider if desired
embedder = rag.Embedder(provider="localllama") # deepinfra")

# create project dir
if not os.path.exists(exampleConfig["location"]):
    os.makedirs(exampleConfig["location"])

# create project
# we use sqlite3 for the example. you might use a different database on the next one
# so we delete the database file first
dbFile = os.sep.join([exampleConfig["location"], exampleConfig["name"]]) + ".db"
print("Creating database:", dbFile)
dbConnectionString = f"sqlite:///{dbFile}"
# delete old file
print("Deleting old DB:", dbConnectionString)
sq.DatabaseUtility.delete_all(dbConnectionString)
# create database tool
db = sq.DatabaseUtility(dbConnectionString)
# create project
prj = sq.Project(
    name=exampleConfig["name"],
    description=exampleConfig["description"],
    langs=exampleConfig["langs"],
    vectorPath=exampleConfig["location"],
    vectorName=f'{exampleConfig["name"]}.vec',
    indexPath=exampleConfig["location"],
    indexName=f'{exampleConfig["name"]}.idx',
    embedModel=embedder.model,
    embedSize=embedder.get_size(),
)

db.insert(prj)
print("Project created:", prj.id)

# set language for rest of example
lang = exampleConfig["langs"].split(",")[0]
# if preview enabled, get LLM
createChunkPreview = exampleConfig.get("chunkPreview",False)
if createChunkPreview:
    llm = {lang:rag.LLM(lang=lang)}


# init tags
tags = []
# init indices
itemIdx = 0
chunkIdx = 0
chunkSize = 100  # use a very small size for example
# iterate items
for i in exampleConfig["items"]:
    print(i["name"])
    # item tag list
    itags = []
    # create tags if new
    for t in i["tags"]:
        if t not in tags:
            tags.append(t)
            tag = sq.Tag(name=t)
            db.insert(tag)
        else:
            tag = db.search(sq.Tag, filters=[sq.Tag.name == t])[0]
        itags.append(tag)
    item = sq.Item(
        itemIdx=itemIdx,
        name=i["name"],
        tags=itags,
        url=i.get("url", ""),
        dataurl=i.get("dataurl", ""),
        imgurl=i.get("imgurl", ""),
    )

    item = db.insert(item)
    print("Item created:", item.id)

    # create snippets
    title = sq.Snippet(
        content=i["title"], lang=lang, itemId=item.id, refIdx=item.itemIdx, type="title"
    )
    title = db.insert(title)

    # optional facts, create empty object if missing
    facts = i.get("facts", {})
    for f in facts.keys():
        fact = sq.Snippet(
            content=json.dumps({f: facts[f]}),
            lang=lang,
            itemId=item.id,
            refIdx=item.itemIdx,
            type="fact",
        )
        db.insert(fact)

    # optional meta, create empty object if missing
    meta = i.get("meta", {})
    metaSnip = sq.Snippet(
        content=json.dumps(meta),
        lang=lang,
        itemId=item.id,
        refIdx=item.itemIdx,
        type="meta",
    )
    db.insert(metaSnip)

    # optional summary. create "" entry if missing
    summary = i.get("text").get(lang).get("summary", "")
    sum = sq.Snippet(
        content=summary,
        lang=lang,
        itemId=item.id,
        refIdx=item.itemIdx,
        type="summary",
    )
    db.insert(sum)

    # get fulltext
    try:
        if i["text"][lang]["selection"] == "file":
            with open(i["text"][lang]["content"]) as f:
                text = f.read()
        else:
            text = i["text"][lang]["content"]
    except:
        raise ("Error reading text")
        exit()

    # insert fulltext
    content = sq.Snippet(
        content=text, lang=lang, itemId=item.id, refIdx=item.itemIdx, type="content"
    )
    content = db.insert(content)

    # create chunks
    # create chunk vectors
    chunks = preprocessor.chunk(text, chunkSize)
    for i, chunk in enumerate(chunks):
        # create item
        print(f"Processing chunk {i}")
        dbChunk = sq.Chunk(itemId=item.id, chunkIdx=chunkIdx)
        # , text = chunk, preview = preview)
        dbChunk = db.insert(dbChunk)

        txt = sq.Snippet(
            content=chunk,
            lang=lang,
            itemId=item.id,
            chunkId=dbChunk.id,
            refIdx=dbChunk.chunkIdx,
            type="content",
        )
        txt = db.insert(txt)
        
        # NB: we could create a chunk summary (preview) here as well
        # would require to use LLM. don't do it here for the test 
        if createChunkPreview:
            preview = llm[lang].preview(chunk)
            pv = sq.Snippet(
                content=preview,
                lang=lang,
                itemId=item.id,
                chunkId=dbChunk.id,
                refIdx=dbChunk.chunkIdx,
                type="summary",
            )
            db.insert(pv)

        chunkIdx += 1

    itemIdx += 1


# print summary
print("Project created:")
print(f"Project: {prj.name}")
print(f"Number of items: {len(db.search(sq.Item))}")
print(f"Number of tags: {len(db.search(sq.Tag))}")
print(f"Number of snippets: {len(db.search(sq.Snippet))}")
print(f"Number of chunks: {len(db.search(sq.Chunk))}")

# create vectors
print("Creating vectors")

prj = db.search(sq.Project)[0]
path = prj.vectorPath
base = prj.vectorName.split(".")[0]
ext = prj.vectorName.split(".")[1]
vecMdl = prj.embedModel
vecSize = prj.embedSize
print(f"Embedder: {vecMdl} - {vecSize}")    

# NB not clear if creating vectors for titles is actually useful, but we do it here
# for demonstration. probably a regular fulltext search using wildcards is more useful
# do not create vectors for fulltext as thie will normally overflow the embedder token limit
for type in ["title","chunk","summary"]:
    collection = f"{base}_{vecSize:04}_{type}_{lang}"
    vector_file = os.sep.join([path,f"{base}_{vecSize:04}_{type}_{lang}.{ext}"])
    print(f"Processing {collection}")
    print(f"Output to {vector_file}")        

    vectors = []
    # we need to process content for chunks and title and summary for items. do not use content as type directly
    if type == "chunk":
        results = db.search(sq.Snippet, filters=[sq.Snippet.lang==lang, sq.Snippet.type=="content", sq.Snippet.chunkId != None],order_by="refIdx")
    else:
        results = db.search(sq.Snippet, filters=[sq.Snippet.lang==lang, sq.Snippet.type==type, sq.Snippet.chunkId == None],order_by="refIdx")
    print(f"Processing {len(results)}, type {type}, lang {lang}")
    for r in results:
        embedding = embedder.encode(r.content)
        vector = np.array(embedding["data"][0]["embedding"]).astype(np.float32)
        # Compute the L2 norm
        norm = np.linalg.norm(vector)

        # Normalize the vector
        if norm != 0:
            normalized_vec = vector / norm
        else:
            normalized_vec = vector  # Handle the zero vector case
            
        vectors.append(normalized_vec.astype(np.float32))

    with open(vector_file, 'wb') as f:
        for vec in vectors:
            f.write(vec.tobytes())
            

