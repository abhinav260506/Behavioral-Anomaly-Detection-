"""
Data Schema Definitions for AI-Powered Behavioral Anomaly Detection System.
Matches the exact schema requirements:
- entity_id
- entity_type (user, service_account, edge_device)
- timestamp
- source_ip / geo_location (lat, lon, country, city)
- resource_accessed
- auth_method (password, token, certificate, biometric)
- session_duration
- command_sequence
- device_fingerprint (OS, MAC address, protocol)
- label (normal / anomaly taxonomy)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class GeoLocation:
    ip: str
    country: str
    city: str
    latitude: float
    longitude: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DeviceFingerprint:
    os_version: str
    mac_address: str
    protocol: str
    browser_or_agent: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AccessLogEvent:
    event_id: str
    entity_id: str
    entity_type: str  # 'user', 'service_account', 'edge_device'
    timestamp: str  # ISO 8601 string
    geo_location: GeoLocation
    resource_accessed: str
    auth_method: str  # 'password', 'token', 'certificate', 'biometric'
    auth_success: bool
    session_duration: float  # seconds
    command_sequence: List[str]
    device_fingerprint: DeviceFingerprint
    data_volume_mb: float = 0.0
    label: str = "normal"  # 'normal', 'brute_force', 'impossible_travel', 'credential_stuffing',
                # 'lateral_movement', 'device_spoofing', 'low_and_slow_exfiltration', 'insider_drift'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

@dataclass
class Alert:
    alert_id: str
    timestamp: str
    entity_id: str
    entity_type: str
    risk_score: float  # 0.0 to 100.0
    is_anomaly: bool
    predicted_attack_type: str
    ground_truth_label: str
    contributing_factors: List[str]
    raw_event: Dict[str, Any]
    explanation_details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
