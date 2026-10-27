"""
Large Synthetic Telemetry Generator Script.
Generates 25,000 to 30,000 realistic sequential access log events with proportional attack campaigns across all 7 attack vectors.
Saves exported files to CSV and JSON in the data/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import random
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import Counter
from typing import List, Dict, Any

from src.generator import SyntheticDataGenerator
from src.schema import AccessLogEvent

def generate_large_dataset(target_count: int = 27500, num_entities: int = 50, seed: int = 42):
    """
    Generates ~27,500 sequential logs with realistic background traffic and repeated attack campaigns.
    """
    print(f"[*] Initializing Synthetic Generator with {num_entities} entities (Seed={seed})...")
    random.seed(seed)
    generator = SyntheticDataGenerator(num_entities=num_entities, seed=seed)

    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    dataset: List[AccessLogEvent] = []

    # Calculate base normal events count (~95-97% of target)
    num_normal = int(target_count * 0.96)
    print(f"[*] Generating {num_normal} normal baseline events over 30 days...")

    curr_time = start_time
    entity_ids = list(generator.entities.keys())

    # Generate sequential benign traffic
    for i in range(num_normal):
        curr_time += timedelta(seconds=random.randint(10, 180))
        uid = random.choice(entity_ids)
        dataset.append(generator.generate_normal_event(uid, curr_time))

    # Inject multiple attack campaigns across all 7 attack vectors
    attack_types = [
        "brute_force",
        "impossible_travel",
        "credential_stuffing",
        "lateral_movement",
        "device_spoofing",
        "low_and_slow_exfiltration",
        "insider_drift"
    ]

    print("[*] Injecting attack campaigns across all 7 attack vectors...")
    # Inject multiple attack instances over the 30-day timeline
    num_campaigns = 15
    for c in range(num_campaigns):
        for atk in attack_types:
            atk_start = start_time + timedelta(days=random.uniform(0.5, 29.5))
            atk_events = generator.generate_attack_sequence(atk, atk_start)
            dataset.extend(atk_events)

    # Sort all events chronologically
    dataset.sort(key=lambda x: x.timestamp)
    total_generated = len(dataset)
    print(f"[+] Total Dataset Generated: {total_generated} events.")

    # Export to CSV and JSON
    os.makedirs("data", exist_ok=True)
    json_path = os.path.abspath("data/synthetic_telemetry_27k.json")
    csv_path = os.path.abspath("data/synthetic_telemetry_27k.csv")

    json_records = []
    csv_rows = []

    for evt in dataset:
        d = evt.to_dict()
        json_records.append(d)
        
        # Flatten structure for CSV
        csv_rows.append({
            "event_id": evt.event_id,
            "entity_id": evt.entity_id,
            "entity_type": evt.entity_type,
            "timestamp": evt.timestamp,
            "ip": evt.geo_location.ip,
            "country": evt.geo_location.country,
            "city": evt.geo_location.city,
            "latitude": evt.geo_location.latitude,
            "longitude": evt.geo_location.longitude,
            "resource_accessed": evt.resource_accessed,
            "auth_method": evt.auth_method,
            "auth_success": evt.auth_success,
            "session_duration": evt.session_duration,
            "commands": "; ".join(evt.command_sequence),
            "os_version": evt.device_fingerprint.os_version,
            "mac_address": evt.device_fingerprint.mac_address,
            "protocol": evt.device_fingerprint.protocol,
            "browser_agent": evt.device_fingerprint.browser_or_agent,
            "label": evt.label
        })

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2)

    # Save CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)

    print(f"    [+] Saved JSON dataset to: {json_path}")
    print(f"    [+] Saved CSV dataset to:  {csv_path}")

    # Calculate statistics breakdown
    counts = Counter(evt.label for evt in dataset)
    stats = []
    for label, count in counts.most_common():
        stats.append({
            "label": label,
            "count": count,
            "pct": round(count / total_generated * 100, 2)
        })

    return {
        "total_generated": total_generated,
        "json_path": json_path,
        "csv_path": csv_path,
        "stats": stats
    }

if __name__ == "__main__":
    res = generate_large_dataset(target_count=27500, num_entities=50, seed=42)
    print("\n" + "="*70)
    print(f"SYNTHETIC DATASET SUMMARY ({res['total_generated']} Total Logs):")
    print("="*70)
    print(f"{'Label / Attack Vector':<30} | {'Exact Count':<12} | {'Percentage':<10}")
    print("-" * 70)
    for st in res["stats"]:
        print(f"{st['label']:<30} | {st['count']:<12} | {st['pct']:<9.2f}%")
    print("="*70)
