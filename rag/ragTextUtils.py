import os
import sys
import re
import nltk
from ragInstrumentation import measure_execution_time
from langdetect import detect, DetectorFactory
# Set deterministic behavior for language detection
DetectorFactory.seed = 0

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
        self.lang = lang
        
    @measure_execution_time
    def clean1(self, text):
        # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces, tabs, and newlines with a single space
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple periods with a single period
        text = text.strip()  # Remove leading and trailing whitespace
        # tokenize the text into sentences
        tokLang = "german" if self.lang == "de" else "english"
        sents_raw = nltk.sent_tokenize(text,language=tokLang)
        sents = []
        wordcount = 0
        for s in sents_raw:
            words = nltk.wordpunct_tokenize(s)
            wordcount += len(words)
            sents.append(" ".join(words).strip())
        # merge the sentences back into a single text            
        # return merged text, wordcount, sentences
        return "".join(sents), wordcount, sents

    @staticmethod
    def remove_unwanted_characters(text):
        """
        Clean the input text by preserving German umlauts and removing unwanted characters.

        :param text: Raw text input.
        :return: Cleaned text.
        """
        # Preserve German umlauts while removing non-printable characters
        allowed_chars = r"[^a-zA-Z0-9äöüÄÖÜß\s.,!?-]"
        text = re.sub(allowed_chars, '', text)

        # Remove overly long words (e.g., gibberish)
        textLen = len(text)
        words = text.split()
        text = [word for word in words if len(word) < 50]
        # check ratio of text len to number of words
        if textLen / len(text) > 10:
            print("Text contains too many long words. Skipping.")
            return None
        text = ' '.join(text)

        return text

    @staticmethod
    def collapse_consecutive_symbols(text):
        """
        Collapse multiple successive newlines, hyphens, or periods into a single one.

        :param text: Input text.
        :return: Cleaned text with collapsed symbols.
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)

        # Replace multiple hyphens with a single one
        text = re.sub(r'-+', '-', text)

        # Replace multiple periods with a single one
        text = re.sub(r'\.+', '.', text)

        # Replace multiple blanks with a single one
        text = re.sub(r'\s+', ' ', text)

        return text

    @measure_execution_time
    def clean(self, text, short=True):
        """
        Preprocess the input text with the following steps:
        1. Remove garbage like non-printable characters, overly long words, etc.
        2. Collapse successive newlines, hyphens, or periods into a single one.
        3. Detect the language of the cleaned text to ensure it's in the allowed list.
        4. Ensure the last sentence ends with a period.

        :param text: Input text to preprocess.
        :return: Cleaned and processed text, or None if text is invalid.
        """
        try:
            # Step 1: Remove garbage
            text = PreProcessor.remove_unwanted_characters(text.strip())
            if not text or (not short and (len(text.split()) < 5)):  # Ensure minimum length
                if DEBUG: print("No text or too short")
                return None,None,None

            # Step 2: Collapse successive symbols
            text = PreProcessor.collapse_consecutive_symbols(text)

            # Step 3: Detect language
            language = detect(text)
            if language != self.lang:
                if DEBUG: print(f"Language detected: {language}")
                return None,None,None

            # Step 4: Ensure the last sentence ends with a period
            if not text.endswith('.'):
                text += '.'

            # Step 5: Tokenize sentences (optional for downstream processing)
            tokLang = "german" if self.lang == "de" else "english"
            sentences = nltk.sent_tokenize(text,language = tokLang)
            wordCnt = len(nltk.wordpunct_tokenize(text))
            return text, wordCnt, sentences

        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            return None,None,None



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
        if wc == None or len(sents) == 1: # nonsense!
            return [""]
        elif wc <= size:
            return [ctext]
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


