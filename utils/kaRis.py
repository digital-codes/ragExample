import requests
#import lxml
import lxml.html as lh
import sys
import os
import json
import mimetypes
import pandas as pd

# http get https://sitzungskalender.karlsruhe.de/db/ratsinformation/termine j==2022 m==0 g== > ratskalender-2022.html

# https://stackoverflow.com/questions/10606133/sending-user-agent-using-requests-library-in-python


yearBase = "https://sitzungskalender.karlsruhe.de/db/ratsinformation/termine"
eventBase = "https://sitzungskalender.karlsruhe.de/db/ratsinformation/"

# create outdir if required
OUTDIR = "out"
try:
    os.mkdir(OUTDIR)
except:
    pass

# #####################
def loadSessions(year):
    """ load sessions for a single year"""
    hdr = headers = {
    'User-Agent': 'Mozilla/5.0'
    #'From': 'youremail@domain.example'  # This is another valid field
    }
    yearUrl = f"{yearBase}?j={year}&m=0"
    print("requesting from ",yearUrl)
    termine = requests.get(yearUrl,headers = hdr)
    if termine.status_code != 200:
        print(f"{yearUrl} failed with {termine.status_code}")
        sys.exit()

    yearHtml = termine.text
    yearDoc = lh.document_fromstring(yearHtml)

    # find all a href ..
    sessionList = []
    aList = yearDoc.findall(".//a")
    for a in aList:
        if not a.attrib.has_key("href"):
            continue
        href = a.attrib["href"]
        if not href.startswith("termin-"):
            continue
        title = a.text
        #print(href, " || ", a.text)
        eventUrl = f"{eventBase}{href}"
        eventFile = f"{OUTDIR}/{year}-{href}.html"
        event = requests.get(eventUrl)
        if event.status_code != 200:
            print(f"{eventUrl} failed with {event.status_code}")
            continue
        sessionList.append({"year":year,"href":href,"title":a.text})
        eventHtml = event.text
        with open(eventFile,"w") as f:
            f.write(eventHtml)

    #print(sessionList)
    with open(f"{year}-sessions.json","w") as f:
        json.dump(sessionList,f)
    return sessionList

# #####################
def loadAttachments(session):
    """ load all attachments for a single session
        returns attachments[],failed[]"""
    attachments = []
    failedLinks = []
    session = s["href"]
    year = s["year"]
    #print(session)
    with open(f"{OUTDIR}/{year}-{session}.html") as f:
        eventHtml = f.read()
    eventDoc = lh.document_fromstring(eventHtml)
    ####### search for assemply type
    assembly = eventDoc.find(".//h3")
    if assembly != None:
        assembly = assembly.text.strip() # normally onle one
    else:
        assembly = "Unknown assembly"
    #######
    tables = eventDoc.findall(".//table")
    eventDate = None
    for t in tables:
        if eventDate != None:
            break
        rows = t.findall(".//tr")
        for r in rows:
            cells = r.findall(".//td")
            if cells[0].text.lower() == "datum":
                eventDate = cells[1].text.strip().split(",")[0]
                break
    #######
    tops = eventDoc.findall(".//id")
    for topIdx in range(1,100):
        topSelector = f"top{topIdx}"
        try:
            top = eventDoc.get_element_by_id(topSelector)
        except KeyError:
            break
        # print(top.attrib," || ",top.text)
        details = top.getnext()
        # print(details.attrib,details.text)
        title = details.find(".//strong")
        if title != None:
            title = title.text.strip()
            #print("Title:",title)
        links = details.findall(".//a")
        for li,l in enumerate(links):
            if not l.attrib.has_key("href"):
                continue
            if not l.attrib.has_key("class"):
                continue
            if l.attrib["class"].lower() != "download-link":
                continue
            linkRef = l.attrib["href"]
            linkName = l.text.lower().strip()
            #print(linkRef,linkName)
            # create descriptor first            
            linkFile = f"{year}-{session}-top{topIdx}-link{li}"
            mimeType = mimetypes.guess_type(linkRef)[0].lower()
            if "image" in mimeType or "application" in mimeType or "text" in mimeType:
                linkFile = ".".join([linkFile,linkRef.split(".")[-1]])
            linkDesc = {"year":year,"session":session,"date":eventDate,
                        "assembly":assembly,"top":topIdx,"topTitle":title,
                        "linkNum":li,"link":linkRef,"title":linkName,"file":linkFile}
            # check path, don't overwrite
            linkPath = f"{OUTDIR}/{linkFile}"
            if not os.path.exists(linkPath):
                # load link
                attachment = requests.get(linkRef)
                if attachment.status_code != 200:
                    print(f"{linkRef} failed with {attachment.status_code}")
                    failedLinks.append({"year":year,"session":session,"link":linkRef,"status":attachment.status_code})
                    continue
                data = attachment.content
                # store attachment
                with open(linkPath,"wb") as f:
                    f.write(data)
##            else:
##                print("Existing: ",linkPath)

            attachments.append(linkDesc)

    return attachments, failedLinks

# #####################

# reread years ...
updateSessions = True 
years = range(2020,2023)

sessions = []
attachments = []
failed = []

# missing file needs update
if not os.path.exists("sessions.json"):
    updateSessions = True

if updateSessions:
    if os.path.exists("sessions.json"):
        with open("sessions.json") as f:
            sessions = json.load(f)

    for y in years:
        print("Year: ",y)
        sessions += loadSessions(y)

    # clean sessions
    df = pd.DataFrame(sessions)
    df.drop_duplicates(subset=["year","href"],keep="last",inplace=True,ignore_index=True)
    df.to_json("sessions.json",orient="records")

##    with open("sessions.json","w") as f:
##        json.dump(sessions,f)

else:
    with open("sessions.json") as f:
        sessions = json.load(f)
        
    
for s in sessions:
    print("Session: ",s["year"],s["href"])
    if int(s["year"]) in years: 
        a,f = loadAttachments(s)
        attachments += a
        failed += f

with open("attachments.json","w") as f:
    json.dump(attachments,f)

with open("failedLinks.json","w") as f:
    json.dump(failed,f)

