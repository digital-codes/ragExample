import numpy as np
from langdetect import detect
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
import ragDeployUtils as rag
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../search')))
import pySearch as ps


# Reference sentences for query types and labels (multilingual)
REFERENCE_INPUTS = {
    "types": {
        "how many": "quantity",
        "who": "subject",
        "which": "subject",
        "when": "date1",
        "last years": "date2",
        "since": "date3",
        "what": "description",
        "where": "location",
        "wie viele": "quantity",
        "wer": "subject",
        "welche": "subject",
        "wann": "date1",
        "in den letzten jahren": "date2",
        "wurde seit": "date3",
        "wurden seit": "date3",
        "was": "description",
        "wo": "location",
    },
    "labels": {
        "refused": "refused",
        "declined": "refused",
        "approved": "approved",
        "accepted": "approved",
        "completed": "completed",
        "decided": "completed",
        "acknowledged": "completed",
        "abgelehnt": "refused",
        "zurückgewiesen": "refused",
        "angenommen": "accepted",
        "beschlossen": "accepted",
        "entschieden": "decided",
        "Beschluß gefasst": "decided",
        "zur Kenntnis genommen": "decided",
    },
}

emb = rag.Embedder()

def detect_language(text):
    """Detect language of input."""
    return detect(text)

def compute_embedding(text):
    """Compute the embedding for a given text."""
    vec = emb.encode(text)["data"][0]["embedding"]
    query_norm = np.linalg.norm(vec) + 1e-9
    query_normalized = vec / query_norm
    return query_normalized

# Flatten the nested dictionary
def create_vectors(items):
    vectors = []
    labels = []
    for key in items.keys():
        value = items[key]
        print("key:value",key,value)
        labels.append(value)
        vectors.append(compute_embedding(key).astype(np.float32)) 
    return labels, vectors

def load_vectors(name,size):
    try:
        arr = ps.load_vectors(f"{name}.vec",size)
        with open(f"{name}.lbl", "r") as f:
        # Read all bytes
            labels = json.load(f)
    except:
        print(f"Error loading {name}")
        raise ValueError(f"Error loading {name}")

    return labels, arr

def process_input(text, thresh=.55):
    """Process user input and match it to query types or labels."""
    lang = detect_language(text)
    if lang not in ["en","de"]:
        raise ValueError(f"Unsupported language: {lang}")

    input_embedding = compute_embedding(text)

    # Match query types and labels
    type_match = ps.query_vectors(type_vectors, input_embedding, 10)
    matches = [t for i,t in enumerate(type_match[0]) if type_match[1][i] > thresh]
    #print("Type match",type_match)
    #for t in type_match[0]:
    type_matches = [(t,type_labels[t]) for t in matches]
    tag_match = ps.query_vectors(tag_vectors, input_embedding, 10)
    #print("Tag match",tag_match)
    matches = [t for i,t in enumerate(tag_match[0]) if tag_match[1][i] > thresh]
    tag_matches = [(t,tag_labels[t]) for t in matches]

    # Determine if input is unrelated
    #is_unrelated = query_type_match is None and not label_matches

    return type_matches, tag_matches


##########
try:
    type_labels, type_vectors = load_vectors("type_vectors_1024",emb.get_size())
    tag_labels, tag_vectors = load_vectors("tag_vectors_1024",emb.get_size())

except:
    print("Creating vectors")
    type_labels, type_vectors = create_vectors(REFERENCE_INPUTS["types"])
    with open("type_vectors_1024.vec", "wb") as f:
        for vec in type_vectors:
            f.write(vec.tobytes())
    with open("type_vectors_1024.lbl", "w") as f:
        json.dump(type_labels,f)
    tag_labels, tag_vectors = create_vectors(REFERENCE_INPUTS["labels"])
    with open("tag_vectors_1024.vec", "wb") as f:
        for vec in tag_vectors:
            f.write(vec.tobytes())
    with open("tag_vectors_1024.lbl", "w") as f:
        json.dump(tag_labels,f)


# Example inputs
inputs = [
    "How many cases were solved in the past 5 years?",
    "Wer hat den Vorschlag akzeptiert?",
    "What is the reason for the delay?",
    "Was ist der Grund für die Verzögerung?",
    "Tell me about the project proposal status.",
    "Beschlüsse zum Klimaschutz in den letzten 5 Jahren",
    "Welche massnahmen wurden seit 2015 umgesetzt",
    "Welche vorhaben wurden seit 2020 beschlossen",
]


# Process each input
for user_input in inputs:
    result = process_input(user_input)
    print(f"Input: {user_input}")
    print(f"Result: {result}\n")
