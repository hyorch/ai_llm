import os.path
import urllib.request
import re # regular expresions

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path = "the-verdict.txt"

if not os.path.exists(file_path):
    print("Download file")
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))

# Tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Preprocessed length:",len(preprocessed))

# Unique Tokens
#all_words = sorted(set(preprocessed)) # set() create a set. Set doesn´t allow duplicates. sorted() crea una lista para añadir orden al set
#vocab_size = len(all_words)
#print("Vocabulari size:",vocab_size)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])


# Vocabulary
#vocab = {token:integer for integer,token in enumerate(all_words)}
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

text = """"Hello !. It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""

tokenizer = SimpleTokenizerV2(vocab)

print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))