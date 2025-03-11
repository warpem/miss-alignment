import json

with open("random_emdb_entries.json") as f:
    d = json.load(f)

titles = [x["admin"]["title"] for x in d]

ribo = [t for t in titles if "ribo" in t.lower()]

print(f"Random EMDB subset contains {len(ribo)} structures with ribo in the title.")

sars = [t for t in titles if "sars" in t.lower()]

print(f"Random EMDB subset contains {len(sars)} structures with sars in the title.")
