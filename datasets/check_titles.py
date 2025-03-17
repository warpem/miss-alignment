import json
from collections import Counter
import string


with open("random_emdb_entries.json") as f:
    d = json.load(f)

titles = [x["admin"]["title"] for x in d]


def count_word_occurrences(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split into words
    words = text.split()

    # Count occurrences
    word_counts = Counter(words)

    return word_counts


# Example usage
text = " ".join([t.strip(".") for t in titles])
word_map = count_word_occurrences(text)

print(word_map)
print(f"Ribosome has {word_map['ribosome']} occurences in the EMDB entry titles.")
print(f"Sarscov2 has {word_map['sarscov2']} occurences in the EMDB entry titles.")
