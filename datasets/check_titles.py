import json
from collections import Counter
import string


with open("random_emdb_entries.json") as f:
    d = json.load(f)

titles = [x["admin"]["title"] for x in d]

ribo = [t for t in titles if "ribo" in t.lower()]

print(f"Random EMDB subset contains {len(ribo)} structures with ribo in the title.")

sars = [t for t in titles if "sars" in t.lower()]

print(f"Random EMDB subset contains {len(sars)} structures with sars in the title.")


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
# Output: Counter({'text': 2, 'this': 2, 'sample': 2, 'words': 2, 'is': 1, 'a': 1, 'contains': 1, 'some': 1, 'that': 1, 'repeat': 1, 'occur': 1, 'multiple': 1, 'times': 1})

# To get the most common words
print(word_map.most_common(3))
# Output: [('text', 2), ('this', 2), ('sample', 2)]
