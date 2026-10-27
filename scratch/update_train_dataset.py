"""
Script to update ONLY the Training Split (first 70%) of data/synthetic_telemetry_27k.csv.
Enhances device_spoofing training scenarios to include 5 legitimate baseline events (h=5)
before spoofing transitions occur, as well as legitimate device migration sequences.
Ensures Validation (15%) and Test (15%) splits remain 100% EXACT AND UNTOUCHED.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint

def generate_enhanced_train_dataset():
    print("="*90)
    print("TASK 1 & TASK 3 — GENERATING ENHANCED TRAINING DATASET WITH MULTI-EVENT HISTORY")
    print("="*90)

    csv_path = "data/synthetic_telemetry_27k.csv"
    df_raw = pd.read_csv(csv_path)

    total_len = len(df_raw)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.85)

    train_df = df_raw.iloc[:train_end].copy()
    val_df = df_raw.iloc[train_end:val_end].copy()
    test_df = df_raw.iloc[val_end:].copy()

    print(f"Original Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # We will enrich device_spoofing entities in train_df
    # Identify device_spoofing entities in train_df
    ds_entities = train_df[train_df["label"] == "device_spoofing"]["entity_id"].unique()
    print(f"Found {len(ds_entities)} device_spoofing entities in Training Split.")

    enhanced_train_rows = []

    # Map entity baseline devices
    entity_base_fps = {}
    for _, r in train_df.iterrows():
        eid = r["entity_id"]
        if eid not in entity_base_fps:
            entity_base_fps[eid] = DeviceFingerprint(
                os_version=r["os_version"],
                mac_address=r["mac_address"],
                protocol=r["protocol"],
                browser_or_agent=r["browser_agent"]
            )

    processed_entities = set()

    for idx, r in train_df.iterrows():
        eid = r["entity_id"]
        lbl = r["label"]

        if lbl == "device_spoofing" and eid not in processed_entities:
            processed_entities.add(eid)

            # Generate multi-event temporal sequence for this entity
            base_fp = entity_base_fps[eid]
            start_t = datetime.fromisoformat(str(r["timestamp"]).replace("Z", "+00:00"))

            # Difficulty 1 (Easy): 5 baseline normal events + 3 rapid spoofing events
            # Difficulty 2 (Medium): 5 baseline normal events + 1 spoofed event + 1 secondary transition
            # Difficulty 3 (Hard): 5 baseline normal events + 1 subtle spoofed event
            diff = random.choice(["easy", "medium", "hard"])

            # 1. Generate 5 Stable Baseline Normal Events
            for b_i in range(5):
                t_b = start_t - timedelta(days=5-b_i, hours=random.randint(0, 4))
                enhanced_train_rows.append({
                    "event_id": f"tr_ds_base_{eid}_{b_i}", "entity_id": eid, "entity_type": r["entity_type"],
                    "timestamp": t_b.isoformat().replace("+00:00", "Z"), "ip": r["ip"], "country": r["country"],
                    "city": r["city"], "latitude": r["latitude"], "longitude": r["longitude"],
                    "resource_accessed": "/dashboard" if b_i < 4 else r["resource_accessed"],
                    "auth_method": r["auth_method"], "auth_success": True,
                    "session_duration": 180.0, "commands": "",
                    "os_version": base_fp.os_version, "mac_address": base_fp.mac_address,
                    "protocol": base_fp.protocol, "browser_agent": base_fp.browser_or_agent,
                    "data_volume_mb": 2.0, "label": "normal"
                })

            # 2. Generate Legitimate Migration Baseline Sequence for 30% of entities (labeled normal)
            if random.random() < 0.30:
                mig_fp = DeviceFingerprint(os_version="Win11 Enterprise", mac_address="00:11:22:33:44:55", protocol="TLSv1.3", browser_or_agent="Chrome/122")
                for m_i in range(3):
                    t_m = start_t - timedelta(hours=12 - m_i*3)
                    enhanced_train_rows.append({
                        "event_id": f"tr_ds_mig_{eid}_{m_i}", "entity_id": eid, "entity_type": r["entity_type"],
                        "timestamp": t_m.isoformat().replace("+00:00", "Z"), "ip": r["ip"], "country": r["country"],
                        "city": r["city"], "latitude": r["latitude"], "longitude": r["longitude"],
                        "resource_accessed": "/dashboard", "auth_method": r["auth_method"], "auth_success": True,
                        "session_duration": 200.0, "commands": "",
                        "os_version": mig_fp.os_version, "mac_address": mig_fp.mac_address,
                        "protocol": mig_fp.protocol, "browser_agent": mig_fp.browser_or_agent,
                        "data_volume_mb": 3.0, "label": "normal"
                    })

            # 3. Generate Spoofed Events (labeled device_spoofing)
            spoof_fp = DeviceFingerprint(
                os_version="Kali Linux 2024.1", mac_address="DE:AD:BE:EF:CA:FE",
                protocol="SSHv2", browser_or_agent="Mozilla/5.0 (Spoofed Linux)"
            )
            t_s1 = start_t + timedelta(minutes=10)
            enhanced_train_rows.append({
                "event_id": f"tr_ds_spoof_{eid}_1", "entity_id": eid, "entity_type": r["entity_type"],
                "timestamp": t_s1.isoformat().replace("+00:00", "Z"), "ip": r["ip"], "country": r["country"],
                "city": r["city"], "latitude": r["latitude"], "longitude": r["longitude"],
                "resource_accessed": r["resource_accessed"], "auth_method": "password", "auth_success": True,
                "session_duration": 120.0, "commands": "nmap -sS" if diff != "hard" else "",
                "os_version": spoof_fp.os_version, "mac_address": spoof_fp.mac_address,
                "protocol": spoof_fp.protocol, "browser_agent": spoof_fp.browser_or_agent,
                "data_volume_mb": 1.5, "label": "device_spoofing"
            })

            if diff in ["easy", "medium"]:
                t_s2 = start_t + timedelta(minutes=25)
                spoof_fp2 = DeviceFingerprint(
                    os_version="Parrot Security OS", mac_address="AA:11:BB:22:CC:33",
                    protocol="SSHv2", browser_or_agent="Agent/1.0"
                )
                enhanced_train_rows.append({
                    "event_id": f"tr_ds_spoof_{eid}_2", "entity_id": eid, "entity_type": r["entity_type"],
                    "timestamp": t_s2.isoformat().replace("+00:00", "Z"), "ip": r["ip"], "country": r["country"],
                    "city": r["city"], "latitude": r["latitude"], "longitude": r["longitude"],
                    "resource_accessed": r["resource_accessed"], "auth_method": "password", "auth_success": True,
                    "session_duration": 150.0, "commands": "mimikatz",
                    "os_version": spoof_fp2.os_version, "mac_address": spoof_fp2.mac_address,
                    "protocol": spoof_fp2.protocol, "browser_agent": spoof_fp2.browser_or_agent,
                    "data_volume_mb": 4.5, "label": "device_spoofing"
                })

        elif lbl != "device_spoofing":
            enhanced_train_rows.append(dict(r))

    new_train_df = pd.DataFrame(enhanced_train_rows)

    # Re-combine into unified dataset: New Train + Untouched Val + Untouched Test
    final_combined_df = pd.concat([new_train_df, val_df, test_df], ignore_index=True)

    output_csv = "data/synthetic_telemetry_27k.csv"
    final_combined_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Successfully updated {output_csv}!")
    print(f"  • New Train Split Rows: {len(new_train_df)}")
    print(f"  • Untouched Validation Split Rows: {len(val_df)}")
    print(f"  • Untouched Test Split Rows: {len(test_df)} (100% Identical!)")

if __name__ == "__main__":
    generate_enhanced_train_dataset()
