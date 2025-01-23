import os
import sys
import re
import nltk
from ragInstrumentation import measure_execution_time

DEBUG = False

# download the punkt tokenizer
if not os.path.exists(nltk.data.find('tokenizers/punkt')):
    nltk.download('punkt', quiet=True)

class PreProcessor():
    def __init__(self,lang="de"):
        nltk.download('punkt_tab')
        #nltk.download('words')
        if lang not in ["de","en"]:
            raise ValueError("Invalid language")
        self.lang = "german" if lang == "de" else "english"

    @measure_execution_time
    def clean(self, text):
        # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces, tabs, and newlines with a single space
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple periods with a single period
        text = text.strip()  # Remove leading and trailing whitespace
        # tokenize the text into sentences
        sents_raw = nltk.sent_tokenize(text,language=self.lang)
        sents = []
        wordcount = 0
        for s in sents_raw:
            words = nltk.wordpunct_tokenize(s)
            wordcount += len(words)
            sents.append(" ".join(words).strip())
        # merge the sentences back into a single text            
        # return merged text, wordcount, sentences
        return "".join(sents), wordcount, sents


    @measure_execution_time
    def chunk(self, text, size=200):
        """
        Split the text into smaller chunks of a fixed size with an overlap.
        
        Args:
            text (str): The text to split.
            chunk_size (int): The size of each chunk (in tokens, not characters).
        Returns:
            list: A list of text chunks.
        """
        if DEBUG: print(f"Chunking text: {text}")
        ctext , wc, sents = self.clean(text)
        if wc <= size:
            return [ctext]
        elif len(sents) == 1: # nonsense!
            return [""]
        else:
            print(f"Chunking {wc} words, {len(sents)} sentences into chunks of {size}")
            chunks = []
            idx = 0
            chunk = ""
            idx1 = 0
            while idx < len(sents):
                print(f"idx: {idx}")
                if DEBUG: print(f"idx: {idx}, chunk: {chunk}")
                chunk = f"{chunk}.{sents[idx]}".strip()
                if len(chunk.split()) >= size:
                    if DEBUG: print(f"Chunk: {chunk},{idx}")
                    if chunk.startswith("."):
                        chunk = chunk[1:]
                    chunks.append(chunk)
                    chunk =  ""
                    idx1 += 1
                    print(f"idx1: {idx1}, {len(chunks)}")
                    if idx1 > 10:
                        break
                    # no idx incremtent here. overlapping chunks
                else:
                    idx += 1
                    idx1 = 0
            return chunks


