"""
Synthetic Data Generator for AI-Powered Behavioral Anomaly Detection.
Simulates realistic user/device access events over time, including normal habitual baselines,
cold-start scenarios, concept drift, and 7 distinct attack patterns.
"""

import random
import uuid
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from src.schema import AccessLogEvent, GeoLocation, DeviceFingerprint

# Static Reference Data for Simulation
CITIES = [
    {"city": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060, "ips": ["198.51.100.10", "198.51.100.11", "198.51.100.12"]},
    {"city": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278, "ips": ["81.2.69.142", "81.2.69.143"]},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "ips": ["202.214.194.1", "202.214.194.2"]},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093, "ips": ["1.120.0.1"]},
    {"city": "Frankfurt", "country": "Germany", "lat": 50.1109, "lon": 8.6821, "ips": ["194.12.200.5"]},
    {"city": "San Francisco", "country": "USA", "lat": 37.7749, "lon": -122.4194, "ips": ["64.233.160.1"]}
]

RESOURCES_BY_ROLE = {
    "finance": ["/api/v1/payroll", "/api/v1/invoices", "/docs/quarterly_report.pdf", "/db/finance_prod"],
    "engineering": ["/git/core-repo", "/api/v1/deploy", "/jira/sprint-board", "/k8s/cluster-admin", "/logs/syslog"],
    "hr": ["/api/v1/employees", "/docs/handbook.pdf", "/portal/benefits"],
    "service_account": ["/api/v1/telemetry", "/metrics", "/healthz", "/db/read_replica"],
    "edge_device": ["/edge/status", "/edge/telemetry_ingest", "/firmware/check"]
}

RESTRICTED_RESOURCES = ["/admin/vault_keys", "/db/master_credentials", "/domain_controller/kerberos", "/exfil/bulk_export"]

AUTH_METHODS = ["password", "token", "certificate", "biometric"]
OS_PROFILES = ["macOS 14.2", "Windows 11 Enterprise", "Ubuntu 22.04 LTS", "iOS 17.1", "Android 14"]
PROTOCOLS = ["TLSv1.3", "SSHv2", "HTTPS", "gRPC"]

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance between two lat/lon points in kilometers."""
    R = 6371.0  # Radius of Earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class SyntheticDataGenerator:
    def __init__(self, num_entities: int = 25, seed: int = 42):
        random.seed(seed)
        self.num_entities = num_entities
        self.entities: Dict[str, Dict] = {}
        self._initialize_entity_baselines()

    def _initialize_entity_baselines(self):
        """Creates realistic habitual profile for each user/service_account/edge_device."""
        roles = ["finance", "engineering", "hr"]
        for i in range(1, self.num_entities + 1):
            if i <= 15:
                entity_type = "user"
                role = roles[i % len(roles)]
                entity_id = f"user_{i:03d}"
            elif i <= 20:
                entity_type = "service_account"
                role = "service_account"
                entity_id = f"svc_acc_{i:03d}"
            else:
                entity_type = "edge_device"
                role = "edge_device"
                entity_id = f"edge_dev_{i:03d}"

            primary_city = random.choice(CITIES)
            os_prof = random.choice(OS_PROFILES)
            mac_addr = f"{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}"
            usual_resources = list(RESOURCES_BY_ROLE[role])
            usual_hours = (8, 18) if entity_type == "user" else (0, 23)

            self.entities[entity_id] = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "role": role,
                "primary_city": primary_city,
                "device_fingerprint": DeviceFingerprint(
                    os_version=os_prof,
                    mac_address=mac_addr,
                    protocol="TLSv1.3" if entity_type != "edge_device" else "gRPC",
                    browser_or_agent="Chrome/120.0" if entity_type == "user" else "DaemonService/2.1"
                ),
                "usual_resources": usual_resources,
                "usual_hours": usual_hours,
                "last_event_time": None,
                "last_geo": None,
                "history_count": 0
            }

    def generate_normal_event(self, entity_id: str, current_time: datetime) -> AccessLogEvent:
        """Generates a benign event adhering to entity profile with minor noise."""
        profile = self.entities[entity_id]
        city = profile["primary_city"]
        
        ip = random.choice(city["ips"])
        geo = GeoLocation(
            ip=ip,
            country=city["country"],
            city=city["city"],
            latitude=city["lat"] + random.uniform(-0.01, 0.01),
            longitude=city["lon"] + random.uniform(-0.01, 0.01)
        )

        resource = random.choice(profile["usual_resources"])
        auth_method = "password" if profile["entity_type"] == "user" else "token"
        
        cmds = []
        if profile["role"] == "engineering" and random.random() < 0.3:
            cmds = ["git pull", "docker build", "kubectl get pods"]

        event = AccessLogEvent(
            event_id=str(uuid.uuid4()),
            entity_id=entity_id,
            entity_type=profile["entity_type"],
            timestamp=current_time.isoformat(),
            geo_location=geo,
            resource_accessed=resource,
            auth_method=auth_method,
            auth_success=True,
            session_duration=float(random.randint(60, 3600)),
            command_sequence=cmds,
            device_fingerprint=profile["device_fingerprint"],
            data_volume_mb=float(round(random.uniform(0.5, 5.0), 2)),
            label="normal"
        )
        
        profile["last_event_time"] = current_time
        profile["last_geo"] = geo
        profile["history_count"] += 1
        return event

    def generate_attack_sequence(self, attack_type: str, start_time: datetime, entity_id: Optional[str] = None) -> List[AccessLogEvent]:
        """Generates a contextual sequence of events representing a specific attack vector."""
        events: List[AccessLogEvent] = []
        target_entity = entity_id if (entity_id and entity_id.strip()) else random.choice(list(self.entities.keys()))

        if target_entity not in self.entities:
            primary_city = random.choice(CITIES)
            self.entities[target_entity] = {
                "entity_id": target_entity,
                "entity_type": "user",
                "role": "engineering",
                "primary_city": primary_city,
                "device_fingerprint": DeviceFingerprint(
                    os_version="Windows 11 Enterprise",
                    mac_address="AA:BB:CC:DD:EE:FF",
                    protocol="TLSv1.3",
                    browser_or_agent="Chrome/120.0"
                ),
                "usual_resources": list(RESOURCES_BY_ROLE["engineering"]),
                "usual_hours": (8, 18),
                "last_event_time": None,
                "last_geo": None,
                "history_count": 0
            }

        profile = self.entities[target_entity]

        if attack_type == "brute_force":
            attacker_city = random.choice(CITIES)
            for i in range(random.randint(12, 18)):
                t = start_time + timedelta(seconds=i * 5)
                events.append(AccessLogEvent(
                    event_id=str(uuid.uuid4()),
                    entity_id=target_entity,
                    entity_type=profile["entity_type"],
                    timestamp=t.isoformat(),
                    geo_location=GeoLocation(
                        ip=attacker_city["ips"][0],
                        country=attacker_city["country"],
                        city=attacker_city["city"],
                        latitude=attacker_city["lat"],
                        longitude=attacker_city["lon"]
                    ),
                    resource_accessed="/api/v1/auth/login",
                    auth_method="password",
                    auth_success=False,
                    session_duration=0.5,
                    command_sequence=[],
                    device_fingerprint=profile["device_fingerprint"],
                    data_volume_mb=0.1,
                    label="brute_force",
                    metadata={"attempt_count": i + 1}
                ))

        elif attack_type == "impossible_travel":
            city1 = profile["primary_city"]
            city2 = [c for c in CITIES if c["city"] != city1["city"]][0]

            t1 = start_time
            t2 = start_time + timedelta(minutes=15)

            events.append(AccessLogEvent(
                event_id=str(uuid.uuid4()),
                entity_id=target_entity,
                entity_type=profile["entity_type"],
                timestamp=t1.isoformat(),
                geo_location=GeoLocation(ip=city1["ips"][0], country=city1["country"], city=city1["city"], latitude=city1["lat"], longitude=city1["lon"]),
                resource_accessed=profile["usual_resources"][0],
                auth_method="token",
                auth_success=True,
                session_duration=300.0,
                command_sequence=[],
                device_fingerprint=profile["device_fingerprint"],
                data_volume_mb=2.5,
                label="normal"
            ))

            events.append(AccessLogEvent(
                event_id=str(uuid.uuid4()),
                entity_id=target_entity,
                entity_type=profile["entity_type"],
                timestamp=t2.isoformat(),
                geo_location=GeoLocation(ip=city2["ips"][0], country=city2["country"], city=city2["city"], latitude=city2["lat"], longitude=city2["lon"]),
                resource_accessed=profile["usual_resources"][0],
                auth_method="password",
                auth_success=True,
                session_duration=120.0,
                command_sequence=[],
                device_fingerprint=profile["device_fingerprint"],
                data_volume_mb=1.2,
                label="impossible_travel",
                metadata={"implausible_speed_kmh": 4500.0}
            ))

        elif attack_type == "credential_stuffing":
            attacker_ip = "185.220.101.5"
            attacker_city = {"city": "Unknown Tor Exit", "country": "Panama", "lat": 8.9824, "lon": -79.5199}
            user_ids = list(self.entities.keys())
            random.shuffle(user_ids)
            
            for i, uid in enumerate(user_ids[:12]):
                t = start_time + timedelta(seconds=i * 8)
                events.append(AccessLogEvent(
                    event_id=str(uuid.uuid4()),
                    entity_id=uid,
                    entity_type=self.entities[uid]["entity_type"],
                    timestamp=t.isoformat(),
                    geo_location=GeoLocation(ip=attacker_ip, country=attacker_city["country"], city=attacker_city["city"], latitude=attacker_city["lat"], longitude=attacker_city["lon"]),
                    resource_accessed="/api/v1/auth/login",
                    auth_method="password",
                    auth_success=(i == 11),
                    session_duration=1.0,
                    command_sequence=[],
                    device_fingerprint=DeviceFingerprint(os_version="Custom Python-urllib", mac_address="00:00:00:00:00:00", protocol="HTTP/1.1", browser_or_agent="Bot/1.0"),
                    data_volume_mb=0.1,
                    label="credential_stuffing"
                ))

        elif attack_type == "lateral_movement":
            # Temporal exploration sequence over time across sensitive endpoints
            t = start_time
            cmds = ["whoami", "netstat -an", "mimikatz.exe", "curl http://internal.vault/dump"]
            for i, res in enumerate(RESTRICTED_RESOURCES + ["/db/finance_prod"]):
                t += timedelta(minutes=random.randint(2, 10))
                events.append(AccessLogEvent(
                    event_id=str(uuid.uuid4()),
                    entity_id=target_entity,
                    entity_type=profile["entity_type"],
                    timestamp=t.isoformat(),
                    geo_location=GeoLocation(ip=profile["primary_city"]["ips"][0], country=profile["primary_city"]["country"], city=profile["primary_city"]["city"], latitude=profile["primary_city"]["lat"], longitude=profile["primary_city"]["lon"]),
                    resource_accessed=res,
                    auth_method="password",
                    auth_success=True,
                    session_duration=600.0,
                    command_sequence=[cmds[i % len(cmds)]],
                    device_fingerprint=profile["device_fingerprint"],
                    data_volume_mb=float(round(random.uniform(5.0, 50.0), 2)),
                    label="lateral_movement"
                ))

        elif attack_type == "device_spoofing":
            # Event 1: Normal baseline event with primary device fingerprint
            t1 = start_time
            events.append(AccessLogEvent(
                event_id=str(uuid.uuid4()),
                entity_id=target_entity,
                entity_type=profile["entity_type"],
                timestamp=t1.isoformat(),
                geo_location=GeoLocation(ip=profile["primary_city"]["ips"][0], country=profile["primary_city"]["country"], city=profile["primary_city"]["city"], latitude=profile["primary_city"]["lat"], longitude=profile["primary_city"]["lon"]),
                resource_accessed=profile["usual_resources"][0],
                auth_method="token",
                auth_success=True,
                session_duration=180.0,
                command_sequence=[],
                device_fingerprint=profile["device_fingerprint"],
                data_volume_mb=2.0,
                label="normal"
            ))

            # Event 2 (10 mins later): Mismatched MAC and OS fingerprint
            t2 = start_time + timedelta(minutes=10)
            spoofed_fp = DeviceFingerprint(
                os_version="Kali Linux 2024.1",
                mac_address="DE:AD:BE:EF:CA:FE",
                protocol="SSHv2",
                browser_or_agent="Mozilla/5.0 (X11; Linux x86_64)"
            )
            events.append(AccessLogEvent(
                event_id=str(uuid.uuid4()),
                entity_id=target_entity,
                entity_type=profile["entity_type"],
                timestamp=t2.isoformat(),
                geo_location=GeoLocation(ip=profile["primary_city"]["ips"][0], country=profile["primary_city"]["country"], city=profile["primary_city"]["city"], latitude=profile["primary_city"]["lat"], longitude=profile["primary_city"]["lon"]),
                resource_accessed=profile["usual_resources"][0],
                auth_method="password",
                auth_success=True,
                session_duration=120.0,
                command_sequence=[],
                device_fingerprint=spoofed_fp,
                data_volume_mb=1.5,
                label="device_spoofing"
            ))

        elif attack_type == "low_and_slow_exfiltration":
            # Multi-day sequence of off-hours (03:15 AM) data transfers accumulating volume
            base_t = start_time.replace(hour=3, minute=15, second=0)
            for day in range(5):
                t = base_t + timedelta(days=day, minutes=random.randint(0, 30))
                events.append(AccessLogEvent(
                    event_id=str(uuid.uuid4()),
                    entity_id=target_entity,
                    entity_type=profile["entity_type"],
                    timestamp=t.isoformat(),
                    geo_location=GeoLocation(ip=profile["primary_city"]["ips"][0], country=profile["primary_city"]["country"], city=profile["primary_city"]["city"], latitude=profile["primary_city"]["lat"], longitude=profile["primary_city"]["lon"]),
                    resource_accessed=RESTRICTED_RESOURCES[day % len(RESTRICTED_RESOURCES)],
                    auth_method="token",
                    auth_success=True,
                    session_duration=45.0,
                    command_sequence=["cat /data/part" + str(day)],
                    device_fingerprint=profile["device_fingerprint"],
                    data_volume_mb=float(round(random.uniform(150.0, 450.0), 2)),
                    label="low_and_slow_exfiltration"
                ))

        elif attack_type == "insider_drift":
            # Multi-day gradual scope expansion accessing new departmental endpoints
            new_endpoints = ["/api/v1/payroll", "/jira/sprint-board", "/api/v1/employees", "/admin/vault_keys", "/api/v1/new_project_alpha"]
            base_t = start_time
            for i, endp in enumerate(new_endpoints):
                t = base_t + timedelta(days=i, hours=random.randint(1, 4))
                events.append(AccessLogEvent(
                    event_id=str(uuid.uuid4()),
                    entity_id=target_entity,
                    entity_type=profile["entity_type"],
                    timestamp=t.isoformat(),
                    geo_location=GeoLocation(ip=profile["primary_city"]["ips"][0], country=profile["primary_city"]["country"], city=profile["primary_city"]["city"], latitude=profile["primary_city"]["lat"], longitude=profile["primary_city"]["lon"]),
                    resource_accessed=endp,
                    auth_method="token",
                    auth_success=True,
                    session_duration=600.0,
                    command_sequence=["git clone " + endp.split("/")[-1]],
                    device_fingerprint=profile["device_fingerprint"],
                    data_volume_mb=float(round(random.uniform(10.0, 80.0), 2)),
                    label="insider_drift"
                ))

        return events

        return events

    def generate_full_dataset(self, num_normal: int = 1500, inject_attacks: bool = True) -> List[AccessLogEvent]:
        """Generates a complete dataset with extreme class imbalance (~98% normal, ~2% anomalous)."""
        dataset: List[AccessLogEvent] = []
        start_time = datetime.now(timezone.utc) - timedelta(days=7)

        # Generate normal background traffic
        curr = start_time
        entity_keys = list(self.entities.keys())

        for _ in range(num_normal):
            curr += timedelta(seconds=random.randint(15, 300))
            uid = random.choice(entity_keys)
            dataset.append(self.generate_normal_event(uid, curr))

        if inject_attacks:
            attack_types = [
                "brute_force",
                "impossible_travel",
                "credential_stuffing",
                "lateral_movement",
                "device_spoofing",
                "low_and_slow_exfiltration",
                "insider_drift"
            ]
            for atk in attack_types:
                atk_start = start_time + timedelta(hours=random.randint(12, 140))
                atk_events = self.generate_attack_sequence(atk, atk_start)
                dataset.extend(atk_events)

        # Sort dataset chronologically
        dataset.sort(key=lambda x: x.timestamp)
        return dataset
