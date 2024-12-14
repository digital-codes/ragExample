import pandas as pd
import os
import sys

# run shell command:
# results = os.popen('ls -l -1 -a').read()

cleanUp = False

imgCmd = "/usr/bin/pdfimages"
#imgParms = "-p  -png -j -jp2 -jbig2 -ccitt -f 1 -l 1000 "
imgParms = "-p  -png -j -f 1 -l 1000 "

# get files with image converstion plugin activated"
# imgFiles = [x.strip() for x in os.popen('grep -l "Image Conversion Plug-in" ./tika/*.json').read().split("\n")]

#Here’s an improved, secure version of the script:
#import subprocess
#import glob
#files = glob.glob('./tika/*.json')

files = os.listdir("./tika")
imgFiles = []
for fl in files:
    if fl == "" or not fl.endswith(".json"):
        continue
    fname = f"./tika/{fl}"
    with open(fname) as file:
        if "Image Conversion Plug-in" in file.read():
            imgFiles.append(fname)
            
#output = subprocess.check_output(['grep', '-l', 'Image Conversion Plug-in'] + files, universal_newlines=True)
#imgFiles = [x.strip() for x in output.split("\n") if x.strip()]

imgPath = "imgs"
if not os.path.exists(imgPath):
    os.mkdir(imgPath)

images = pd.DataFrame(columns=["dir","file","size"])

print("Img files: ",len(imgFiles))


fileList = []

for f in imgFiles:
    # remove first dir level and extension
    if f == "":
        continue
    baseName = f.split("/")[2].split(".pdf.json")[0]
    # input files are in "./out"
    fileList.append(baseName)
    

# also get files larger than 10MB. Suspect to contain images
largeFiles = [x.strip() for x in os.popen('find out/*.pdf -size +1M').read().split("\n")]

for f in largeFiles:
    if f == "":
        continue
    baseName = f.split("/")[1].split(".pdf")[0]
    if not baseName in fileList: 
        fileList.append(baseName)
    else:
        print("Exists:",baseName)

skipFiles = [
    "2015-termin-3845-top2-link5",
    "2016-termin-4132-top16-link1",
    "2016-termin-4142-top5-link19",
    "2016-termin-4146-top17-link1",
    "2016-termin-4146-top7-link3",
    "2015-termin-3833-top2-link1",
    "2015-termin-3833-top3-link1",
    "2015-termin-3835-top5-link2",
    "2015-termin-3845-top3-link1",
    "2016-termin-4125-top13-link2",
    "2015-termin-3844-top2-link9"
    ]

for f in fileList:    
    if f == "":
        continue
    fName = "/".join(["out",".".join([f,"pdf"])])
    imgBase = "/".join([".",imgPath,f])
    # some files don't work ..
    if any(term in f for term in skipFiles):
        continue
    
    #print(outDir)
    # remove old
    if not cleanUp:
        os.system(f"rm -rf {imgBase}*")
    # extract
    # extract needs trailing / on out dir
    #extractCmd = f"{imgCmd} {imgParms} {fName} {outDir}/"
    extractCmd = f"{imgCmd} {imgParms} {fName} {imgBase}"
    # print(extractCmd)
    try:
        if not cleanUp:
            os.system(extractCmd)
    except:
        print("Failed on ",f)
    # some pdfs produce  large number of small files
    # check and delete
    # using anything like ls * or rm * fails on very large subset
    # due to max_args (see xargs, getconfig)
    # iterate of page and check individual files
    for pi in range (1,1000):
        x = os.popen(f"ls {imgPath}/{f}-{pi:03}-000.*").read()
        #print(f"{imgPath}/{f}-{pi:03}-000.* : {x}")
        if x == "":
            break
        # try to find image 20 per page
        fn = f"{imgPath}/{f}-{pi:03}-020.*"
        fnn = os.popen(f"ls {fn}").read().split("\n")
        if len(fnn) > 0 and fnn[-1] != "":
            print("Deleting ...",fn)
            for ii in range(0,1000000):
                rmcmd = f"rm {imgPath}/{f}-{pi:03}-{ii:03}.*"
                if os.system(rmcmd) != 0:
                    print("failed on ",rmcmd)
                    #sys.exit()
                    break
                else:
                    print(f"{rmcmd} OK")
                
    
# finally delete small files
os.system('find imgs/ -size -1000c -exec rm {} \;')


for i in os.listdir(imgPath):
    images = images.append({"dir":imgPath,"file":i,"size":os.stat("/".join([imgPath,i])).st_size},ignore_index=True)

images.to_json("images.json",orient="records")
