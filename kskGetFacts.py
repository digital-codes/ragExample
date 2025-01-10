import os
import numpy as np
import ragDeployUtils as deployUtils

import ragSqlUtils as sq
import private_remote as pr

import json

DEBUG = False

llm_de = deployUtils.Llm(lang="de")
llm_en = deployUtils.Llm(lang="en")

connString = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
db = sq.DatabaseUtility(connString)
# read project data
prj = db.search(sq.Project)[0]
print(f"Project: {prj.name}")


itemTexts = db.search(sq.Snippet,filters = [sq.Snippet.lang=="de",sq.Snippet.type=="content",sq.Snippet.chunkId == None])
print(f"Items: {len(itemTexts)}")
for text in itemTexts:
    itemId = text.itemId
    facts = db.search(sq.Snippet,filters = [sq.Snippet.lang=="de",sq.Snippet.type=="fact",sq.Snippet.itemId == itemId])
    if len(facts) > 0:
        # skip existing facts
        print(f"Item: {itemId} already has {len(facts)} facts")
        continue
    itemIdx = text.refIdx
    tx = text.content.strip()
    print(f"item: {itemId},{itemIdx}",)
    #print(f"{tx}")
    try:
        wrappedFacts = llm_de.getFacts(tx)[0].replace("`","")
        print(f"WrappedFacts: {wrappedFacts}")
        # we have a leading "json" sometimes that we need to remove
        # so make sure we remove everything before the first [
        # or before the first { and extract element 0 if []
        if "[" in wrappedFacts:
            encodedFacts = wrappedFacts.split("[")[-1].split("]")[0].strip()
            #encodedFacts = encodedFacts[encodedFacts.find("["):]
        else:
            encodedFacts = wrappedFacts[wrappedFacts.find("{"):wrappedFacts.find("}")+1].strip()
        facts = json.loads(f"{encodedFacts}")
        print(f"Facts: {len(facts)},{facts}")
    except Exception as e:
        print(f"Error: {e},{wrappedFacts},{encodedFacts}")
        exit()
        continue

    for fact in facts:
        print(f"Fact: {fact}")
        try:
            db.insert(sq.Snippet(itemId=itemId,refIdx=itemIdx,lang="de",type="fact",content=json.dumps(fact)))
        except Exception as e:
            print(f"Error: {e}")
            continue
    

