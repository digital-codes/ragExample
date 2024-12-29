import ragSqlUtils as sq
import private_remote as pr
import os
import json

cs = f'mysql+pymysql://{pr.mysql["user"]}:{pr.mysql["password"]}@{pr.mysql["host"]}:{pr.mysql["port"]}/{pr.mysql["database"]}'
db = sq.DatabaseUtility(cs)

projects = [p.name for p in db.search(sq.Project)]

print("Available projects:",projects)

name = input("Select project: ")

sel = db.search(sq.Project,filters = [sq.Project.name == name])
if len(sel) == 0:
    print("Invalid selection")
    exit()

prId = sel[0].id
path = sel[0].vectorPath
base = sel[0].vectorName
print("ID:",prId)

# load summary file
sf = input("Summary file: ")

try:
    with open(sf) as f:
        summary = json.load(f)
except:
    print("Invalid file",sf)
    exit()

for s in summary:
    meta = json.loads(s["meta"])
    name = f'{meta["area"]}_{meta["bundle"]}_{meta["topic"]}'
    print(name)
    item = db.search(sq.Item,filters = [sq.Item.name == name])[0]
    print(item.id)
    item.summary = s["text"]
    db.update(sq.Item,item)


# chunks
#v = db.get_chunk_vectors(prId)
