# see https://textmining.wp.hs-hannover.de/Terminologie.html
import glob
import nltk
from nltk.corpus import stopwords
import codecs
import treetaggerwrapper
from collections import Counter
import re
import math
import pprint
import json

########
refdist = nltk.FreqDist()
nrOfWordsRef = 0

freqdata = codecs.open("./DeReKo-2014-II-MainArchive-STT.100000.freq", "r", "utf-8")
for line in freqdata:
    (word,lemma,pos,freq) = line.split('\t')
    nrOfWordsRef+= float(freq)
    refdist[lemma] = float(freq)
    
freqdata.close()

#########
tagger = treetaggerwrapper.TreeTagger(TAGLANG='de',TAGDIR="/opt/install/naturalLanguage/treetagger/")
fdist = Counter() 
filelist = glob.glob("./textout/20*.txt")

nrOfWords = 0

######
stops = set(stopwords.words('german'))
stops.add("@card@")
stops.add("in+die")
stops.add("☐")
stops.add("ja")
stops.add("nein")
stops.add("bzw")
stops.add("bzw.")



def stopTokenize(sent):
    words = nltk.word_tokenize(sent,language='german')
    words_filtered = []
    for w in words:
        if w.lower() not in stops:
            if len(w) > 2:
                words_filtered.append(w)
    # print(" ".join(words_filtered))            
    return words_filtered

###### 
h_mark = re.compile(r'==+')
leerzeilen = re.compile(r'\n+')

for datei in filelist:
    try:
        textfile = codecs.open(datei, "r", "utf-8")
    except:
        continue
    text = textfile.read()
    textfile.close()
    if len(text) < 100:
        continue
    
    #text = h_mark.sub('\n\n',text)
    leerzeilen = re.compile(r'\n+')
    paragraphs = leerzeilen.split(text)
    for par in paragraphs:
        sentences = nltk.sent_tokenize(par,language='german')
        #sentences_tok = [nltk.word_tokenize(sent,language='german') for sent in sentences]
        sentences_tok = [stopTokenize(sent) for sent in sentences]
    
        for sent in sentences_tok:
            tags = tagger.tag_text(sent,tagonly=True) 
            nrOfWords += len(tags)
            words = [tag.lemma for tag in treetaggerwrapper.make_tags(tags) if not tag.pos[0] == "$"]
            fdist.update(words)
            
#c = refdist['es']/fdist['es']
c = 1

#word_rel_freq = {lemma:math.log(c * fdist.get(lemma) / refdist.get(lemma,1)) for lemma in fdist }
#word_rel_freq = sorted(word_rel_freq.items(), key=lambda x: x[1],reverse=True)
#pprint.pprint(word_rel_freq[0:100])
word_freq = sorted(fdist.items(), key=lambda x: x[1],reverse=True)
pprint.pprint(word_freq[0:100])
with open("distribution.json","w") as f:
    json.dump(word_freq,f)
    
