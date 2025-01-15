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
        #print(f"WrappedFacts: {wrappedFacts}")
        # we have a leading "json" sometimes that we need to remove
        # so make sure we remove everything before the first [
        # or before the first { and extract element 0 if []
        if "[" in wrappedFacts:
            encodedFacts = wrappedFacts[wrappedFacts.find("["):wrappedFacts.find("]")+1].strip()
            #encodedFacts = wrappedFacts.split("[")[-1].split("]")[0].strip()
            #encodedFacts = encodedFacts[encodedFacts.find("["):]
            facts_ = json.loads(f"{encodedFacts}")
            facts = []
            for fact in facts_:
                if 'category' in fact and 'fact' in fact:
                    facts.append({fact['category']: fact['fact']})
                else:
                    facts.append(fact)
        else:
            encodedFacts = wrappedFacts[wrappedFacts.find("{"):wrappedFacts.find("}")+1].strip()
            facts_ = json.loads(f"{encodedFacts}")
            facts = []
            for fact in facts_:
                facts.append({fact:facts_[fact]})
        #facts = json.loads(f"{encodedFacts}")
        print(f"Facts: {len(facts)},{facts}")
        # iterate over fact and merge same category into one
        mergedFacts = []
        for fact in facts:
            key = list(fact.keys())[0]
            found = False
            for mergedFact in mergedFacts:
                if key in mergedFact:
                    mergedFact[key] = (f"{mergedFact[key]}. {fact[key]}")
                    found = True
                    break
            if not found:
                mergedFacts.append({key:fact[key]})
    except Exception as e:
        print(f"Error: {e},{wrappedFacts},{encodedFacts},{mergedFacts}")
        continue

    for fact in mergedFacts:
        text = json.dumps(fact)
        try:
            #db.insert(sq.Snippet(itemId=itemId,refIdx=itemIdx,lang="de",type="fact",content=json.dumps(fact)))
            db.insert(sq.Snippet(itemId=itemId,refIdx=itemIdx,lang="de",type="fact",content=text))
        except Exception as e:
            print(f"Error: {e}: {fact}")
            continue
    

