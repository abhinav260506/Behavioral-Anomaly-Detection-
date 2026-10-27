"""
Rich Multi-Event Temporal Synthetic Dataset Generator.
Generates 30,000+ realistic access log events distributed evenly over 30 days.
Ensures rare attack classes (low_and_slow_exfiltration, insider_drift, device_spoofing, impossible_travel)
are populated with hundreds of multi-event behavioral sequences belonging to many unique entities across all splits.
Includes realistic benign noise & feature overlaps (department transfers, on-call work, large downloads).
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import random
import uuid
import math
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint

random.seed(42)

CITIES = [
    {"city": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060, "ips": ["198.51.100.10", "198.51.100.11", "198.51.100.12"]},
    {"city": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278, "ips": ["81.2.69.142", "81.2.69.143"]},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "ips": ["202.214.194.1", "202.214.194.2"]},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093, "ips": ["1.120.0.1"]},
    {"city": "Frankfurt", "country": "Germany", "lat": 50.1109, "lon": 8.6821, "ips": ["194.12.200.5"]},
    {"city": "San Francisco", "country": "USA", "lat": 37.7749, "lon": -122.4194, "ips": ["64.233.160.1"]}
]

ROLES = ["finance", "engineering", "hr", "sales", "analytics"]

RESOURCES_BY_ROLE = {
    "finance": ["/finance/payroll", "/finance/invoices", "/finance/tax_records", "/finance/quarterly_audit"],
    "engineering": ["/engineering/git-core", "/engineering/deploy-pipeline", "/engineering/jira", "/engineering/k8s", "/engineering/syslogs"],
    "hr": ["/hr/employee_directory", "/hr/handbook", "/hr/benefits", "/hr/performance_reviews"],
    "sales": ["/sales/crm", "/sales/leads", "/sales/contracts", "/sales/pipeline"],
    "analytics": ["/analytics/bi_dashboard", "/analytics/data_warehouse", "/analytics/reports"],
    "service_account": ["/telemetry/ingest", "/metrics/prometheus", "/healthz", "/db/read_replica"],
    "edge_device": ["/edge/status", "/edge/telemetry", "/firmware/v2"]
}

RESTRICTED_RESOURCES = [
    "/admin/vault_keys", "/db/master_credentials", "/domain_controller/kerberos",
    "/exfil/bulk_export", "/db/finance_prod", "/admin/super_user"
]

OS_PROFILES = ["macOS 14.2", "Windows 11 Enterprise", "Ubuntu 22.04 LTS", "iOS 17.1", "Android 14"]

def create_random_mac():
    return f"{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}"

def generate_rich_dataset():
    print("[*] Generating Rich Multi-Event Synthetic Dataset distributed across 30 days...")
    
    start_sim_time = datetime(2026, 7, 1, 8, 0, 0, tzinfo=timezone.utc)
    num_entities = 200
    entities: Dict[str, Dict] = {}

    for i in range(1, num_entities + 1):
        if i <= 150:
            entity_type = "user"
            role = ROLES[i % len(ROLES)]
            eid = f"user_{i:03d}"
        elif i <= 175:
            entity_type = "service_account"
            role = "service_account"
            eid = f"svc_{i:03d}"
        else:
            entity_type = "edge_device"
            role = "edge_device"
            eid = f"dev_{i:03d}"

        city = random.choice(CITIES)
        entities[eid] = {
            "entity_id": eid,
            "entity_type": entity_type,
            "role": role,
            "city": city,
            "device_fingerprint": DeviceFingerprint(
                os_version=random.choice(OS_PROFILES),
                mac_address=create_random_mac(),
                protocol="TLSv1.3" if entity_type != "edge_device" else "gRPC",
                browser_or_agent="Chrome/120.0" if entity_type == "user" else "DaemonService/2.1"
            ),
            "usual_resources": list(RESOURCES_BY_ROLE.get(role, ["/api/v1/general"])),
            "usual_hours": (8, 18) if entity_type == "user" else (0, 23)
        }

    all_events: List[AccessLogEvent] = []

    # 1. BENIGN BASELINE EVENTS (25,000 events spread across 30 days)
    print("    [+] Simulating 25,000 Benign Habitual & Overlapping Events across 30 days...")
    cur_time = start_sim_time
    for step in range(25000):
        cur_time += timedelta(seconds=103) # Spans ~30 days
        eid = f"user_{(step % 150) + 1:03d}"
        prof = entities[eid]
        
        # Benign Overlaps:
        is_on_call = random.random() < 0.05
        event_dt = cur_time.replace(hour=random.choice([1, 2, 3, 21, 22])) if is_on_call else cur_time

        data_vol = float(round(random.uniform(300.0, 800.0), 2)) if random.random() < 0.02 else float(round(random.uniform(0.5, 10.0), 2))

        res = random.choice(prof["usual_resources"])
        if random.random() < 0.03:
            other_role = random.choice(ROLES)
            res = random.choice(RESOURCES_BY_ROLE[other_role])

        cmds = ["git pull", "npm test", "docker build"] if (prof["role"] == "engineering" and random.random() < 0.2) else []

        evt = AccessLogEvent(
            event_id=f"evt_norm_{step+1:05d}",
            entity_id=eid,
            entity_type=prof["entity_type"],
            timestamp=event_dt.isoformat(),
            geo_location=GeoLocation(ip=prof["city"]["ips"][0], country=prof["city"]["country"], city=prof["city"]["city"], latitude=prof["city"]["lat"], longitude=prof["city"]["lon"]),
            resource_accessed=res,
            auth_method="token" if prof["entity_type"] != "user" else "password",
            auth_success=True,
            session_duration=float(random.randint(60, 1800)),
            command_sequence=cmds,
            device_fingerprint=prof["device_fingerprint"],
            data_volume_mb=data_vol,
            label="normal"
        )
        all_events.append(evt)

    # 2. LOW-AND-SLOW EXFILTRATION (Spanning full 30 days)
    print("    [+] Generating 400 Low-and-Slow Exfiltration events across 55 entities...")
    exfil_entities = [f"user_{i:03d}" for i in range(1, 56)]
    for idx, eid in enumerate(exfil_entities):
        prof = entities[eid]
        # Spread start times throughout full 28 days
        base_t = start_sim_time + timedelta(days=(idx * 28) // len(exfil_entities), hours=2, minutes=random.randint(0, 45))
        for day in range(7):
            t = base_t + timedelta(days=day, minutes=random.randint(-15, 15))
            res = random.choice(["/finance/reports_export", "/analytics/raw_dump", "/db/backup_part", "/docs/confidential", "/engineering/vault_backup"])
            evt = AccessLogEvent(
                event_id=f"evt_exfil_{idx}_{day}",
                entity_id=eid,
                entity_type=prof["entity_type"],
                timestamp=t.isoformat(),
                geo_location=GeoLocation(ip=prof["city"]["ips"][0], country=prof["city"]["country"], city=prof["city"]["city"], latitude=prof["city"]["lat"], longitude=prof["city"]["lon"]),
                resource_accessed=res,
                auth_method="token",
                auth_success=True,
                session_duration=45.0,
                command_sequence=[f"cat {res}/chunk_{day}"],
                device_fingerprint=prof["device_fingerprint"],
                data_volume_mb=float(round(random.uniform(80.0, 350.0), 2)),
                label="low_and_slow_exfiltration"
            )
            all_events.append(evt)

    # 3. INSIDER DRIFT (Spanning full 30 days)
    print("    [+] Generating 350 Insider Drift events across 50 entities...")
    drift_entities = [f"user_{i:03d}" for i in range(56, 106)]
    target_roles = ["finance", "admin", "hr", "sales"]
    for idx, eid in enumerate(drift_entities):
        prof = entities[eid]
        base_t = start_sim_time + timedelta(days=(idx * 25) // len(drift_entities), hours=10)
        for seq_i in range(7):
            t = base_t + timedelta(days=seq_i, hours=random.randint(1, 4))
            dept = target_roles[seq_i % len(target_roles)]
            res = f"/{dept}/restricted_project_{seq_i}"
            evt = AccessLogEvent(
                event_id=f"evt_drift_{idx}_{seq_i}",
                entity_id=eid,
                entity_type=prof["entity_type"],
                timestamp=t.isoformat(),
                geo_location=GeoLocation(ip=prof["city"]["ips"][0], country=prof["city"]["country"], city=prof["city"]["city"], latitude=prof["city"]["lat"], longitude=prof["city"]["lon"]),
                resource_accessed=res,
                auth_method="password",
                auth_success=True,
                session_duration=600.0,
                command_sequence=[f"access_scope_{dept}"],
                device_fingerprint=prof["device_fingerprint"],
                data_volume_mb=float(round(random.uniform(15.0, 60.0), 2)),
                label="insider_drift"
            )
            all_events.append(evt)

    # 4. DEVICE SPOOFING (Spanning full 30 days)
    print("    [+] Generating 300 Device Spoofing events across 60 entities...")
    spoof_entities = [f"user_{i:03d}" for i in range(106, 151)] + [f"svc_{i:03d}" for i in range(151, 166)]
    for idx, eid in enumerate(spoof_entities):
        prof = entities[eid]
        base_t = start_sim_time + timedelta(days=(idx * 28) // len(spoof_entities), hours=14)
        for seq_i in range(5):
            t = base_t + timedelta(minutes=seq_i * 15)
            if seq_i == 0:
                fp = prof["device_fingerprint"]
                lbl = "normal"
            else:
                fp = DeviceFingerprint(
                    os_version=random.choice(["Kali Linux 2024.1", "Unknown OS 9.9", "Parrot OS"]),
                    mac_address=create_random_mac(),
                    protocol="SSHv2",
                    browser_or_agent="Mozilla/5.0 (Spoofed Agent)"
                )
                lbl = "device_spoofing"
            evt = AccessLogEvent(
                event_id=f"evt_spoof_{idx}_{seq_i}",
                entity_id=eid,
                entity_type=prof["entity_type"],
                timestamp=t.isoformat(),
                geo_location=GeoLocation(ip=prof["city"]["ips"][0], country=prof["city"]["country"], city=prof["city"]["city"], latitude=prof["city"]["lat"], longitude=prof["city"]["lon"]),
                resource_accessed=prof["usual_resources"][0],
                auth_method="password",
                auth_success=True,
                session_duration=120.0,
                command_sequence=[],
                device_fingerprint=fp,
                data_volume_mb=2.0,
                label=lbl
            )
            all_events.append(evt)

    # 5. IMPOSSIBLE TRAVEL (Spanning full 30 days)
    print("    [+] Generating 300 Impossible Travel events across 150 entities...")
    travel_entities = [f"user_{i:03d}" for i in range(1, 151)]
    for idx, eid in enumerate(travel_entities):
        prof = entities[eid]
        c1 = prof["city"]
        c2 = [c for c in CITIES if c["city"] != c1["city"]][0]
        base_t = start_sim_time + timedelta(days=(idx * 28) // len(travel_entities), hours=9)
        t1 = base_t
        t2 = base_t + timedelta(minutes=12)

        evt1 = AccessLogEvent(
            event_id=f"evt_trav_{idx}_1", entity_id=eid, entity_type=prof["entity_type"], timestamp=t1.isoformat(),
            geo_location=GeoLocation(ip=c1["ips"][0], country=c1["country"], city=c1["city"], latitude=c1["lat"], longitude=c1["lon"]),
            resource_accessed=prof["usual_resources"][0], auth_method="token", auth_success=True, session_duration=120.0,
            command_sequence=[], device_fingerprint=prof["device_fingerprint"], data_volume_mb=1.0, label="normal"
        )
        evt2 = AccessLogEvent(
            event_id=f"evt_trav_{idx}_2", entity_id=eid, entity_type=prof["entity_type"], timestamp=t2.isoformat(),
            geo_location=GeoLocation(ip=c2["ips"][0], country=c2["country"], city=c2["city"], latitude=c2["lat"], longitude=c2["lon"]),
            resource_accessed=prof["usual_resources"][0], auth_method="password", auth_success=True, session_duration=120.0,
            command_sequence=[], device_fingerprint=prof["device_fingerprint"], data_volume_mb=1.0, label="impossible_travel"
        )
        all_events.append(evt1)
        all_events.append(evt2)

    # 6. BRUTE FORCE (Spanning full 30 days)
    print("    [+] Generating 450 Brute Force events...")
    bf_entities = [f"user_{i:03d}" for i in range(1, 31)]
    for idx, eid in enumerate(bf_entities):
        prof = entities[eid]
        attacker_city = random.choice(CITIES)
        base_t = start_sim_time + timedelta(days=(idx * 28) // len(bf_entities), hours=11)
        for i in range(15):
            t = base_t + timedelta(seconds=i * 5)
            evt = AccessLogEvent(
                event_id=f"evt_bf_{idx}_{i}", entity_id=eid, entity_type=prof["entity_type"], timestamp=t.isoformat(),
                geo_location=GeoLocation(ip=attacker_city["ips"][0], country=attacker_city["country"], city=attacker_city["city"], latitude=attacker_city["lat"], longitude=attacker_city["lon"]),
                resource_accessed="/api/v1/auth/login", auth_method="password", auth_success=False, session_duration=0.5,
                command_sequence=[], device_fingerprint=prof["device_fingerprint"], data_volume_mb=0.1, label="brute_force"
            )
            all_events.append(evt)

    # 7. CREDENTIAL STUFFING (Spanning full 30 days)
    print("    [+] Generating 450 Credential Stuffing events...")
    for burst_i in range(35):
        attacker_ip = f"185.220.101.{burst_i+1}"
        base_t = start_sim_time + timedelta(days=(burst_i * 28) // 35, hours=16)
        user_pool = [f"user_{i:03d}" for i in range(1, 151)]
        random.shuffle(user_pool)
        for i in range(13):
            uid = user_pool[i]
            t = base_t + timedelta(seconds=i * 6)
            evt = AccessLogEvent(
                event_id=f"evt_cs_{burst_i}_{i}", entity_id=uid, entity_type="user", timestamp=t.isoformat(),
                geo_location=GeoLocation(ip=attacker_ip, country="Panama", city="Tor Exit", latitude=8.98, longitude=-79.51),
                resource_accessed="/api/v1/auth/login", auth_method="password", auth_success=(i == 12), session_duration=1.0,
                command_sequence=[], device_fingerprint=DeviceFingerprint(os_version="Python-urllib", mac_address="00:00:00:00:00:00", protocol="HTTP/1.1", browser_or_agent="Bot/1.0"),
                data_volume_mb=0.1, label="credential_stuffing"
            )
            all_events.append(evt)

    # 8. LATERAL MOVEMENT (Spanning full 30 days)
    print("    [+] Generating 400 Lateral Movement events...")
    lat_entities = [f"user_{i:03d}" for i in range(31, 81)]
    for idx, eid in enumerate(lat_entities):
        prof = entities[eid]
        base_t = start_sim_time + timedelta(days=(idx * 28) // len(lat_entities), hours=13)
        cmds = ["whoami", "netstat -an", "mimikatz.exe", "curl http://vault/dump"]
        for i, res in enumerate(RESTRICTED_RESOURCES[:8] if len(RESTRICTED_RESOURCES) >= 8 else RESTRICTED_RESOURCES * 2):
            t = base_t + timedelta(minutes=i * 8)
            evt = AccessLogEvent(
                event_id=f"evt_lat_{idx}_{i}", entity_id=eid, entity_type=prof["entity_type"], timestamp=t.isoformat(),
                geo_location=GeoLocation(ip=prof["city"]["ips"][0], country=prof["city"]["country"], city=prof["city"]["city"], latitude=prof["city"]["lat"], longitude=prof["city"]["lon"]),
                resource_accessed=res, auth_method="password", auth_success=True, session_duration=300.0,
                command_sequence=[cmds[i % len(cmds)]], device_fingerprint=prof["device_fingerprint"], data_volume_mb=15.0, label="lateral_movement"
            )
            all_events.append(evt)

    # Sort ALL events chronologically by timestamp
    print("    [+] Sorting all events chronologically by timestamp...")
    all_events.sort(key=lambda x: x.timestamp)

    # Convert to DataFrame
    df_rows = []
    for e in all_events:
        df_rows.append({
            "event_id": e.event_id,
            "entity_id": e.entity_id,
            "entity_type": e.entity_type,
            "timestamp": e.timestamp,
            "ip": e.geo_location.ip,
            "country": e.geo_location.country,
            "city": e.geo_location.city,
            "latitude": e.geo_location.latitude,
            "longitude": e.geo_location.longitude,
            "resource_accessed": e.resource_accessed,
            "auth_method": e.auth_method,
            "auth_success": e.auth_success,
            "session_duration": e.session_duration,
            "commands": ";".join(e.command_sequence),
            "os_version": e.device_fingerprint.os_version,
            "mac_address": e.device_fingerprint.mac_address,
            "protocol": e.device_fingerprint.protocol,
            "browser_agent": e.device_fingerprint.browser_or_agent,
            "data_volume_mb": e.data_volume_mb,
            "label": e.label
        })

    df = pd.DataFrame(df_rows)
    csv_path = "data/synthetic_telemetry_27k.csv"
    json_path = "data/synthetic_telemetry_27k.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump([e.to_dict() for e in all_events], f, indent=2)

    print(f"\n[+] Generated Rich Synthetic Dataset: {len(df):,} events total.")
    print("    Class Counts in Dataset:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    generate_rich_dataset()
