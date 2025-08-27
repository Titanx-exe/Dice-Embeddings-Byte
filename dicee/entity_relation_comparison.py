#!/usr/bin/env python3
import os
from collections import Counter

def parse_file(path):
    heads, rels, tails = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                print(f"Warning: skipping malformed line {lineno} in {path!r}: {line!r}")
                continue
            h, r, t = parts
            heads.append(h)
            rels.append(r)
            tails.append(t)
    return heads, rels, tails

def analyze(train_path, valid_path, test_path, list_new_items=False):
    for p in (train_path, valid_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    # Parse files
    train_heads, train_rels, train_tails = parse_file(train_path)
    valid_heads, valid_rels, valid_tails = parse_file(valid_path)
    test_heads, test_rels, test_tails = parse_file(test_path)

    # Entity and relation sets
    train_entities = set(train_heads) | set(train_tails)
    valid_entities = set(valid_heads) | set(valid_tails)
    test_entities  = set(test_heads)  | set(test_tails)

    train_relations = set(train_rels)
    valid_relations = set(valid_rels)
    test_relations  = set(test_rels)

    # New items
    new_valid_entities     = valid_entities - train_entities
    new_test_entities      = test_entities  - train_entities
    new_test_entities_full = test_entities  - (train_entities | valid_entities)

    new_valid_relations = valid_relations - train_relations
    new_test_relations  = test_relations  - train_relations

    # Counters
    cnt_train_heads = Counter(train_heads)
    cnt_train_rels  = Counter(train_rels)
    cnt_train_tails = Counter(train_tails)
    cnt_valid_heads = Counter(valid_heads)
    cnt_valid_rels  = Counter(valid_rels)
    cnt_valid_tails = Counter(valid_tails)
    cnt_test_heads  = Counter(test_heads)
    cnt_test_rels   = Counter(test_rels)
    cnt_test_tails  = Counter(test_tails)

    def print_section(title):
        print(f"\n=== {title} ===")

    print_section("File sizes (number of triples)")
    print(f"Train: {len(train_heads)}")
    print(f"Valid: {len(valid_heads)}")
    print(f"Test : {len(test_heads)}")

    print_section("Unique counts per file")
    print(f"Train: unique heads={len(set(train_heads))}, relations={len(train_relations)}, tails={len(set(train_tails))}, total entities={len(train_entities)}")
    print(f"Valid: unique heads={len(set(valid_heads))}, relations={len(valid_relations)}, tails={len(set(valid_tails))}, total entities={len(valid_entities)}")
    print(f"Test : unique heads={len(set(test_heads))}, relations={len(test_relations)}, tails={len(set(test_tails))}, total entities={len(test_entities)}")

    print_section("New items in valid vs train")
    print(f"New entities in valid: {len(new_valid_entities)}")
    print(f"New relations in valid: {len(new_valid_relations)}")
    if list_new_items:
        print("\nNew entities in valid:")
        for e in sorted(new_valid_entities): print(f"  {e}")
        print("\nNew relations in valid:")
        for r in sorted(new_valid_relations): print(f"  {r}")

    print_section("New items in test vs train")
    print(f"New entities in test: {len(new_test_entities)}")
    print(f"New relations in test: {len(new_test_relations)}")
    if list_new_items:
        print("\nNew entities in test:")
        for e in sorted(new_test_entities): print(f"  {e}")
        print("\nNew relations in test:")
        for r in sorted(new_test_relations): print(f"  {r}")

    print_section("Unique test entities NOT in train or valid")
    print(f"Test entities not in train or valid: {len(new_test_entities_full)}")
    if list_new_items:
        print("\nCompletely new entities in test:")
        for e in sorted(new_test_entities_full): print(f"  {e}")

    # Most frequent items in train
    def print_top(counter, name, top_n=10):
        print(f"Top {top_n} {name}:")
        for item, count in counter.most_common(top_n):
            print(f"  {item}: {count}")

    print_section("Top-10 frequent items in train")
    print_top(cnt_train_heads, "heads")
    print_top(cnt_train_rels,  "relations")
    print_top(cnt_train_tails, "tails")

if __name__ == "__main__":
    # Your hardcoded paths
    train_file = "C://Users//Harshit Purohit//Byte//myenv6//Lib//site-packages//dicee//KGs//UMLS//train.txt"
    valid_file = "C://Users//Harshit Purohit//Byte//myenv6//Lib//site-packages//dicee//KGs//UMLS//valid.txt"
    test_file  = "C://Users//Harshit Purohit//Byte//myenv6//Lib//site-packages//dicee//KGs//UMLS//test.txt"

    # Set to True if you want to print lists of new entities/relations
    list_new = False

    analyze(train_file, valid_file, test_file, list_new_items=list_new)




# Usage
train_file = ("C://Users//Harshit Purohit//Byte//myenv6//Lib//site-packages//dicee//KGs//YAGO3-10//train.txt")
valid_file = ("C://Users//Harshit Purohit//Byte//myenv6//Lib//site-packages//dicee//KGs//YAGO3-10//valid.txt")
test_file = ("C://Users//Harshit Purohit//Byte//myenv6//Lib//site-packages//dicee//KGs//YAGO3-10//test.txt")
# analyze_kg_splits(train_file, valid_file, test_file)
