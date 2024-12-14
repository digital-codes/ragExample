# before running this script, you need to start tika server
# java -jar tika-server-standard-2.9.2.jar --port 8081
# and export env as follows:
# export TIKA_SERVER_ENDPOINT="localhost:8081"

from tika import parser
import os
import json
files = os.listdir("./out")
for f in files:
    if f.endswith("pdf"):
        parsed = parser.from_file(os.sep.join(["out",f]))
        fout = os.sep.join(["tika",f.split(".pdf")[0]+".json"])
        with open(fout,"w") as fo:
            json.dump(parsed,fo)
        # fout = os.sep.join(["tika",f.split(".pdf")[0]+".txt"])
        # with open(fout,"w") as fo:
        #     text_content = parsed['content']
        #     fo.write(text_content)

