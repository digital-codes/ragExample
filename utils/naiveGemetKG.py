import rdflib
import nltk
import re
import json

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../rag')))
import ragDeployUtils as rag
embedder = rag.Embedder(provider="localllama")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../search')))
import pySearch as ps

gemetVectors = ps.load_vectors("gemet_labels_de.vec", embedder.get_size())
with open("gemet_labels_de_keys.json") as f:
    gemetKeys = json.load(f)


with open("gemet_labels_de.json") as f:
    lbls = json.load(f)

#1 
# Load GEMET graph from your partial/fixed files (German labels, backbone, etc.)
gemet_graph = rdflib.Graph()
gemet_graph.parse("gemet_merged.rdf", format="xml")  # etc.

# Collect German labels in a dict: { label_text_lower: concept_uri }
gemet_labels_de = {}

SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")

for s, p, o in gemet_graph.triples((None, RDFS.label, None)):
    # Check if the label is German
    if getattr(o, 'language', None) == 'de':
        label_str = o.lower()
        gemet_labels_de[label_str] = s  # map text -> concept URI

# Also gather skos:prefLabel if any are used in German
for s, p, o in gemet_graph.triples((None, SKOS.prefLabel, None)):
    if getattr(o, 'language', None) == 'de':
        label_str = o.lower()
        gemet_labels_de[label_str] = s


print("Loaded", len(gemet_labels_de), "German labels from GEMET")
output_file = "gemet_labels_de.rdf"
with open(output_file, "w") as f:
    json.dump(gemet_labels_de, f)


# 2
def preprocess(text):
    # Basic cleaning: lowercase, remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def tokenize_and_filter(text):
    tokens = word_tokenize(text, language='german')
    german_stops = set(stopwords.words('german'))
    # remove stopwords, keep only alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and t not in german_stops]
    return tokens

#3 
def find_gemet_concepts(tokens, gemet_labels):
    found_concepts = []
    # Join tokens into bigrams/trigrams if needed or keep as single tokens
    # For simplicity, let's do single-token exact matches here.
    for i,t in enumerate(tokens):
        tv = embedder.encode(t)["data"][0]["embedding"]
        result = ps.query_vectors(gemetVectors, tv, 1)
        score = result[1][0]
        if score > 0.75:
            idx = result[0][0]
            key = gemetKeys[idx]
            uri = gemet_labels.get(key, None)
            #print("Adding concept:", key,uri,idx,score)
            found_concepts.append((key, uri))
            # replace token with key. otherwise relation extraction will not work
            tokens[i] = key
    return found_concepts

# 4 
def extract_simple_relations(tokens, concept_hits):
    """
    Look for pattern: <concept> verursacht <concept>
    Return list of (subject_URI, 'verursacht', object_URI) 
    """
    # Convert concept_hits to a dict: index -> concept_uri
    # concept_hits is a list of (token, concept_uri)
    # We need the index in tokens to match them
    index_map = {}
    for i, tok in enumerate(tokens):
        for ch_tok, ch_uri in concept_hits:
            if tok == ch_tok:
                index_map[i] = ch_uri

    relations = []
    for i, tok in enumerate(tokens):
        if tok == 'verursacht':
            # Check if there's a concept to the left and to the right
            left_uri = index_map.get(i-1, None)
            right_uri = index_map.get(i+1, None)
            if left_uri and right_uri:
                relations.append((left_uri, "http://example.org/ontology/verursacht", right_uri))
    return relations


# 5
text = """
Die Luftverschmutzung verursacht Klimawandel. Erneuerbare Energien sind wichtig für den Klimaschutz.
"""

sentences = sent_tokenize(text, language='german')

all_relations = []

for sent in sentences:
    print("\n\nProcessing sentence:", sent)
    cleaned = preprocess(sent)
    tokens = tokenize_and_filter(cleaned)
    concept_hits = find_gemet_concepts(tokens, gemet_labels_de)
    # e.g., concept_hits might include [('klimawandel', URIRef('...'))] if that label matches
    rels = extract_simple_relations(tokens, concept_hits)
    all_relations.extend(rels)

#print("Extracted concepts:", concept_hits)
#print("Extracted relations:", all_relations)
# Example output: [(URIRef('...Luftverschmutzung...'), "http://example.org/ontology/verursacht", URIRef('...Klimawandel...'))]


# 6
kg = rdflib.Graph()

for (subj, pred, obj) in all_relations:
    kg.add((subj, rdflib.URIRef(pred), obj))

# Optional: store the text snippet or sentence as well (provenance)
for i, (subj, pred, obj) in enumerate(all_relations, start=1):
    # e.g., attach a blank node with the sentence text
    provenance_bnode = rdflib.BNode()
    kg.add((provenance_bnode, rdflib.URIRef("http://purl.org/dc/terms/source"), rdflib.Literal(text)))
    kg.add((provenance_bnode, rdflib.URIRef("http://example.org/ontology/relationSubject"), subj))
    kg.add((provenance_bnode, rdflib.URIRef("http://example.org/ontology/relationPredicate"), rdflib.URIRef(pred)))
    kg.add((provenance_bnode, rdflib.URIRef("http://example.org/ontology/relationObject"), obj))

# Serialize the KG in Turtle format
print(kg.serialize(format='xml')) #.decode('utf-8'))

