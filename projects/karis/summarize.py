"""summarize files into a summary file"""

import json
import os
import datetime
import pandas as pd

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rag")))

import private_remote as pr

import ragTextUtils as textUtils
import ragDeployUtils as deployUtils
import ragSqlUtils as sql
import ragConfig as config


DEBUG = False


# text stuff
preprocessor = textUtils.PreProcessor()

llmProvider = "openai"
#model = "llama3.2:latest" # config.ollama["lngMdl"]
#model = config.ollama["lngMdl"]
model = "granite3.3:8b" # config.ollama["lngMdl"]
url = config.ollama["lngUrl"]

# get model
llm = deployUtils.Llm(provider=llmProvider, model=model)
llm.url = url

# open database
dbname = "/mnt_ai/sqlite/karis.db"
connection_string = f"sqlite:///{dbname}"
db = sql.DatabaseUtility(connection_string)

contentSnips = db.search(
    sql.Snippet,
    filters=[
        sql.Snippet.lang == "de",
        sql.Snippet.type == "content",
        sql.Snippet.chunkId == None,
    ],
)

summarySnips = db.search(
    sql.Snippet,
    filters=[
        sql.Snippet.lang == "de",
        sql.Snippet.type == "summary",
        sql.Snippet.chunkId == None,
    ],
)

itemIds = [s.itemId for s in contentSnips]
print(f"items: {len(itemIds)}")
for i,snip in enumerate(contentSnips):
    try:
        text = llm.summarize(snip.content,size=300)[0].strip()
        sum = next((s for s in summarySnips if s.itemId == snip.itemId), None)
        if not sum:
            print(f"no summary for {snip.itemId}")
            continue
        print(f"itemId: {snip.itemId}, {sum.itemId}")
        sum.content = text
        db.update(sql.Snippet,sum)
        print(f"updated {i}")
    except Exception as e:
        print(f"error {i}: {e}")
        pass
