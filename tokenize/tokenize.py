import os.path
import urllib.request
import re # regular expresions

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
all_words = sorted(set(preprocessed)) # set() create a set. Set doesn´t allow duplicates. sorted() crea una lista para añadir orden al set
vocab_size = len(all_words)
print("Vocabulari size:",vocab_size)

# Vocabulary
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 10:
        break