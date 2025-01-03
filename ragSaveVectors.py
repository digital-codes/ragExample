import ragSqlUtils as sq
import private_remote as pr
import os

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

try:
    os.mkdir(path)
except FileExistsError:
    pass

# chunks
v = db.get_chunk_vectors(prId)
exit()
fn = os.sep.join([path,f"chunk_{base}"])
with open (fn,"wb") as f:
    size = f.write(v.astype("float32"))
print("Chunk vector written:",fn,size)
    
# titles
v = db.get_title_vectors(prId)
fn = os.sep.join([path,f"title_{base}"])
with open (fn,"wb") as f:
    size = f.write(v.astype("float32"))
print("Title vectors written:",fn,size)
    
