import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
from ragInstrumentation import measure_execution_time
import ragSqlUtils as sq
import ragDeployUtils as rag
import ragTextUtils as rt
import json

exampleConfigFile = "example.json"

try:
    with open(exampleConfigFile) as f:
        exampleConfig = json.load(f)
except Exception as e:      
    print(f"Error reading {exampleConfigFile}: {e}")
    sys.exit()

# text stuff
preprocessor = rt.PreProcessor()
# get embedder. use different provider if desired
embedder = rag.Embedder(provider="deepinfra")

# create project
# we use sqlite3 for the example. you might use a different database on the next one
# so we delete the database file first
dbFile = os.sep.join([exampleConfig["location"],exampleConfig["name"]]) + ".db"
print("Creating database:",dbFile)
dbConnectionString = f"sqlite:///{dbFile}"
# delete old file
print("Deleting old DB:",dbConnectionString)
sq.DatabaseUtility.delete_all(dbConnectionString)
# create database tool
db = sq.DatabaseUtility(dbConnectionString)
# create project
prj = sq.Project(
    name = exampleConfig["name"], description=exampleConfig["description"], langs = exampleConfig["langs"],
    vectorPath = exampleConfig["location"], vectorName = f'{exampleConfig["name"]}.vec', 
    indexPath = exampleConfig["location"], indexName = f'{exampleConfig["name"]}.idx', 
    embedModel=embedder.model, embedSize=embedder.get_size()
    )

db.insert(prj)
print("Project created:",prj.id)

# set language for rest of example
lang = exampleConfig["langs"].split(",")[0]


# init tags
tags = []
# init indices
itemIdx = 0
chunkIdx = 0
chunkSize = 100 # use a very small size for example
# iterate items
for i in exampleConfig["items"]:
    print(i["name"])
    # item tag list
    itags = []
    # create tags if new
    for t in i["tags"]:
        if t not in tags:
            tags.append(t)
            tag = sq.Tag(name = t)
            db.insert(tag)
        else:
            tag = db.search(sq.Tag, filters = [sq.Tag.name == t])[0]
        itags.append(tag)
    item = sq.Item(itemIdx = itemIdx, name = i["name"],tags = itags,
         url = i.get("url",""), dataurl = i.get("dataurl",""), imgurl = i.get("imgurl","")
         )

    item = db.insert(item)
    print("Item created:",item.id)

    # create snippets
    title = sq.Snippet(content=i["title"], lang=lang, itemId = item.id, refIdx = item.itemIdx, type="title")
    title = db.insert(title)

    # meta/facts
    meta = i.get("meta",{})
    metaFact = sq.Snippet(content=json.dumps({"meta":meta}), lang=lang, itemId = item.id, refIdx = item.itemIdx, type="fact")
    metaFact = db.insert(metaFact)

    # check fulltext
    try:
        if i["text"][lang]["selection"] == "file":
            with open (i["text"][lang]["content"]) as f:
                text = f.read()
        else:
            text = i["text"][lang]["content"]
    except:
        raise("Error reading text")
        exit()

    # insert fulltext
    content = sq.Snippet(content=text, lang=lang, itemId = item.id, refIdx = item.itemIdx, type="content")
    content = db.insert(content)

    # create chunks
    # create chunk vectors
    chunks = preprocessor.chunk(text,chunkSize)
    for i, chunk in enumerate(chunks):
        # create item
        print(f"Processing chunk {i}")
        dbChunk = sq.Chunk(itemId = item.id, chunkIdx = chunkIdx)
        # , text = chunk, preview = preview)
        dbChunk = db.insert(dbChunk)

        txt = sq.Snippet(content=chunk, lang=lang, itemId = item.id, chunkId = dbChunk.id, refIdx = dbChunk.chunkIdx, type="content")
        txt = db.insert(txt)


        chunkIdx += 1 

    itemIdx += 1                

