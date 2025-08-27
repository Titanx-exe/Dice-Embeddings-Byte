import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
# ... (other imports)

class PreprocessKG:
    # ... (existing code)

    def get_all_entities_and_relations(self):
        """
        DEFINITIVE FIX:
        This function is modified to get all unique entities and relations
        from the train, validation, AND test sets to create a complete vocabulary.
        This prevents all downstream errors.
        """
        print("Building complete vocabulary from all data splits (train, valid, test)...")
        
        # 1. Use pandas to concatenate all subject/object and predicate columns.
        all_entities = pd.concat([
            self.kg.raw_train_set.iloc[:, 0], self.kg.raw_train_set.iloc[:, 2],
            self.kg.raw_valid_set.iloc[:, 0], self.kg.raw_valid_set.iloc[:, 2],
            self.kg.raw_test_set.iloc[:, 0],  self.kg.raw_test_set.iloc[:, 2]
        ]).unique()

        all_relations = pd.concat([
            self.kg.raw_train_set.iloc[:, 1],
            self.kg.raw_valid_set.iloc[:, 1],
            self.kg.raw_test_set.iloc[:, 1]
        ]).unique()

        # 2. Assign the complete, sorted lists to the KG object.
        self.kg.entities = sorted(list(all_entities))
        self.kg.relations = sorted(list(all_relations))

        self.kg.num_entities = len(self.kg.entities)
        self.kg.num_relations = len(self.kg.relations)
        self.kg.entity_to_idx = {e: i for i, e in enumerate(self.kg.entities)}
        self.kg.relation_to_idx = {r: i for i, r in enumerate(self.kg.relations)}
        print(f"Vocabulary built. Total unique entities: {self.kg.num_entities}, Total unique relations: {self.kg.num_relations}")


    def _add_reciprocal_triples(self):
        # ... (rest of the file is unchanged)