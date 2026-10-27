import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from collections import Counter
from src.generator import SyntheticDataGenerator
from src.train import DatasetTrainer

def calculate_attack_distribution(num_logs=2000, seed=42):
    generator = SyntheticDataGenerator(num_entities=30, seed=seed)
    full_dataset = generator.generate_full_dataset(num_normal=num_logs, inject_attacks=True)

    trainer = DatasetTrainer(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_set, val_set, test_set = trainer.split_dataset(full_dataset)

    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "full": full_dataset
    }

    # Count attack types in each split
    stats = {}
    total_attacks_overall = sum(1 for e in full_dataset if e.label != "normal")

    all_labels = sorted(list(set(e.label for e in full_dataset)))

    for split_name, data in splits.items():
        counts = Counter(e.label for e in data)
        total_events = len(data)
        attacks_in_split = sum(count for label, count in counts.items() if label != "normal")
        
        stats[split_name] = {
            "total_events": total_events,
            "total_attacks": attacks_in_split,
            "pct_of_total_attacks_in_dataset": round(attacks_in_split / total_attacks_overall * 100, 2) if total_attacks_overall > 0 else 0,
            "attack_breakdown": {}
        }
        
        for label in all_labels:
            if label == "normal":
                continue
            cnt = counts[label]
            # pct of this attack type vs total attacks in this split
            pct_in_split_attacks = round(cnt / attacks_in_split * 100, 2) if attacks_in_split > 0 else 0
            # pct of this attack type vs total attacks overall in dataset
            pct_of_all_attacks = round(cnt / total_attacks_overall * 100, 2) if total_attacks_overall > 0 else 0
            
            stats[split_name]["attack_breakdown"][label] = {
                "count": cnt,
                "pct_within_split_attacks": pct_in_split_attacks,
                "pct_of_all_attacks_overall": pct_of_all_attacks
            }

    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    calculate_attack_distribution()
