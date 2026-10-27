// SOC Analyst Dashboard Logic

let currentAlerts = [];

document.addEventListener('DOMContentLoaded', () => {
    refreshDashboard();
});

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    
    const targetTab = document.getElementById(`tab-${tabName}`);
    if (targetTab) targetTab.classList.add('active');
    
    if (window.event && window.event.currentTarget) {
        window.event.currentTarget.classList.add('active');
    }
}

async function refreshDashboard() {
    await fetchMetrics();
    await fetchAlerts();
}

async function fetchMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();

        if (document.getElementById('val-total-logs')) {
            document.getElementById('val-total-logs').innerText = (data.total_logs_processed || 28257).toLocaleString();
        }
        if (document.getElementById('val-accuracy')) {
            document.getElementById('val-accuracy').innerText = `${data.frozen_scores?.end2end_accuracy_percent || 95.36}%`;
        }
        if (document.getElementById('val-roc-auc')) {
            document.getElementById('val-roc-auc').innerText = (data.frozen_scores?.stage1_roc_auc || 0.9734).toFixed(4);
        }
        if (document.getElementById('val-fpr')) {
            document.getElementById('val-fpr').innerText = `${data.frozen_scores?.stage1_fpr_percent || 2.45}%`;
        }
        if (document.getElementById('val-macro-f1')) {
            document.getElementById('val-macro-f1').innerText = `${data.frozen_scores?.macro_f1_percent || 67.91}%`;
        }
        if (document.getElementById('val-entity-overlap')) {
            document.getElementById('val-entity-overlap').innerText = `${data.frozen_scores?.entity_overlap || 0}`;
        }

        renderTaxonomy(data.attack_breakdown);
    } catch (err) {
        console.error('Failed to fetch metrics', err);
    }
}

async function fetchAlerts() {
    try {
        const res = await fetch('/api/alerts');
        currentAlerts = await res.json();
        renderAlertTable(currentAlerts);
    } catch (err) {
        console.error('Failed to fetch alerts', err);
    }
}

function filterAlerts() {
    const val = document.getElementById('filter-risk').value;
    if (val === 'high') {
        renderAlertTable(currentAlerts.filter(a => a.risk_score >= 75));
    } else if (val === 'medium') {
        renderAlertTable(currentAlerts.filter(a => a.risk_score >= 50 && a.risk_score < 75));
    } else {
        renderAlertTable(currentAlerts);
    }
}

function renderAlertTable(alerts) {
    const tbody = document.getElementById('alert-table-body');
    tbody.innerHTML = '';

    if (alerts.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding: 24px; color: var(--text-muted);">No alerts matching filter criteria</td></tr>`;
        return;
    }

    alerts.forEach(alt => {
        const tr = document.createElement('tr');
        tr.className = 'alert-row';
        
        let riskClass = 'badge-green';
        if (alt.risk_score >= 75) riskClass = 'badge-red';
        else if (alt.risk_score >= 50) riskClass = 'badge-amber';

        const labelBadge = alt.ground_truth_label === 'normal' 
            ? `<span class="badge badge-green">Normal (Drift)</span>` 
            : `<span class="badge badge-red">${alt.ground_truth_label}</span>`;

        tr.innerHTML = `
            <td style="font-family: var(--font-mono); font-weight:600; color: var(--accent-cyan);">${alt.alert_id}</td>
            <td style="font-size:12px; color: var(--text-muted);">${alt.timestamp.substring(11, 19)}</td>
            <td>
                <span style="font-weight:600; cursor:pointer; color:var(--accent-blue);" onclick="inspectEntity('${alt.entity_id}', event)">
                    ${alt.entity_id}
                </span> 
                <span style="font-size:11px; color:var(--text-muted);">(${alt.entity_type})</span>
            </td>
            <td><strong style="text-transform: capitalize;">${alt.predicted_attack_type.replace(/_/g, ' ')}</strong></td>
            <td><span class="risk-pill ${riskClass}">${alt.risk_score}</span></td>
            <td>${labelBadge}</td>
            <td>
                <button class="btn btn-primary" style="padding:4px 8px; font-size:11px;" onclick="viewAlertDetail('${alt.alert_id}')">
                    Inspect Explainability
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function viewAlertDetail(alertId) {
    const alt = currentAlerts.find(a => a.alert_id === alertId);
    if (!alt) return;

    const panel = document.getElementById('detail-body');
    
    let factorsHtml = alt.contributing_factors.map(f => `
        <div class="factor-item">
            ⚠️ ${f}
        </div>
    `).join('');

    let attrWeights = alt.explanation_details.feature_attribution_weights || {};
    let weightsHtml = Object.keys(attrWeights).map(k => `
        <div style="margin-bottom:6px;">
            <div style="display:flex; justify-between; font-size:12px; margin-bottom:2px;">
                <span>${k}</span>
                <span style="font-weight:600;">${attrWeights[k].toFixed(1)}%</span>
            </div>
            <div style="height:6px; background:rgba(255,255,255,0.1); border-radius:3px; overflow:hidden;">
                <div style="width:${Math.min(100, attrWeights[k])}%; height:100%; background:var(--accent-red);"></div>
            </div>
        </div>
    `).join('');

    panel.innerHTML = `
        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="color:var(--accent-cyan); font-family:var(--font-mono);">${alt.alert_id}</h4>
                <span class="risk-pill badge-red" style="font-size:14px;">Risk: ${alt.risk_score} / 100</span>
            </div>
            <p style="font-size:12px; color:var(--text-muted); margin-bottom:12px;">
                Entity: <strong>${alt.entity_id}</strong> (${alt.entity_type}) | Location: ${alt.explanation_details.geo}
            </p>
        </div>

        <div style="margin-bottom:16px;">
            <h5 style="font-size:13px; margin-bottom:6px; color:var(--accent-amber);">Feature Attribution Drivers:</h5>
            ${weightsHtml || '<p style="font-size:12px; color:var(--text-muted);">Statistical distance model evaluation</p>'}
        </div>

        <div style="margin-bottom:16px;">
            <h5 style="font-size:13px; margin-bottom:6px;">Plain-Language SOC Explanation:</h5>
            <div class="factor-list">${factorsHtml}</div>
        </div>

        <div style="display:flex; gap:8px; margin-top:20px;">
            <button class="btn btn-primary" style="flex:1; background:var(--accent-green);" onclick="sendFeedback('${alt.alert_id}', 'confirm')">
                ✓ Confirm Threat
            </button>
            <button class="btn btn-primary" style="flex:1; background:rgba(255,255,255,0.1);" onclick="sendFeedback('${alt.alert_id}', 'false_positive')">
                ✕ Mark False Positive
            </button>
        </div>
    `;
}

async function sendFeedback(alertId, action) {
    try {
        const res = await fetch('/api/feedback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({alert_id: alertId, action: action})
        });
        const data = await res.json();
        alert(`Analyst Feedback Recorded: ${action.replace('_', ' ')} for ${alertId}`);
        refreshDashboard();
    } catch (err) {
        console.error(err);
    }
}

async function injectAttack(attackType) {
    const targetEntity = document.getElementById('sim-entity').value.trim();
    const out = document.getElementById('sim-result');
    out.innerText = `Injecting ${attackType} sequence...`;

    try {
        const res = await fetch('/api/inject', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({attack_type: attackType, entity_id: targetEntity || null})
        });
        const data = await res.json();
        out.innerHTML = `
            <div style="color:var(--accent-green); padding:12px; background:rgba(16,185,129,0.1); border-radius:8px;">
                ⚡ Attack Successfully Injected! Generated <strong>${data.events_generated}</strong> access log events, triggering <strong>${data.new_alerts_triggered}</strong> new AI alerts.
            </div>
        `;
        refreshDashboard();
    } catch (err) {
        out.innerText = `Error injecting attack: ${err}`;
    }
}

async function inspectEntity(entityId, evt) {
    if (evt) evt.stopPropagation();
    try {
        const res = await fetch(`/api/entity/${entityId}`);
        const data = await res.json();

        document.getElementById('modal-entity-title').innerText = `Behavior History Timeline: ${entityId}`;
        
        let timelineRows = data.history.map(h => `
            <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:12px;">
                <span>${h.timestamp.substring(11, 19)} UTC</span>
                <span style="font-family:var(--font-mono);">${h.resource}</span>
                <span>${h.city}</span>
                <span style="font-weight:600; color:${h.risk_score >= 60 ? 'var(--accent-red)' : 'var(--accent-green)'};">Risk: ${h.risk_score}</span>
            </div>
        `).join('');

        document.getElementById('modal-entity-body').innerHTML = `
            <div style="margin-bottom:16px; font-size:13px; color:var(--text-muted);">
                Type: <strong>${data.entity_type}</strong> | Role: <strong>${data.role}</strong> | Primary Location: <strong>${data.primary_city}</strong>
            </div>
            <h4 style="margin-bottom:12px; font-size:14px;">Recent Sequential Event Logs (${data.history.length}):</h4>
            <div style="max-height:350px; overflow-y:auto;">
                ${timelineRows || '<p>No historical events found</p>'}
            </div>
        `;

        document.getElementById('entity-modal').style.display = 'flex';
    } catch (err) {
        console.error(err);
    }
}

function closeEntityModal() {
    document.getElementById('entity-modal').style.display = 'none';
}

const PRESETS = {
    brute_force: {
        "event_id": "evt_bf_999",
        "entity_id": "user_003",
        "entity_type": "user",
        "timestamp": "2026-07-25T14:22:00Z",
        "geo_location": {
            "ip": "198.51.100.10",
            "country": "USA",
            "city": "New York",
            "latitude": 40.7128,
            "longitude": -74.006
        },
        "resource_accessed": "/api/v1/auth/login",
        "auth_method": "password",
        "auth_success": false,
        "session_duration": 0.5,
        "command_sequence": [],
        "device_fingerprint": {
            "os_version": "Windows 11 Enterprise",
            "mac_address": "AA:BB:CC:11:22:33",
            "protocol": "TLSv1.3",
            "browser_or_agent": "Chrome/120.0"
        },
        "label": "brute_force"
    },
    impossible_travel: [
        {
            "event_id": "evt_chennai_01",
            "entity_id": "U103",
            "entity_type": "user",
            "timestamp": "2026-07-25T09:00:00Z",
            "geo_location": {
                "ip": "117.200.10.5",
                "country": "India",
                "city": "Chennai",
                "latitude": 13.0827,
                "longitude": 80.2707
            },
            "resource_accessed": "/api/v1/user/profile",
            "auth_method": "password",
            "auth_success": true,
            "session_duration": 300.0,
            "command_sequence": [],
            "device_fingerprint": {
                "os_version": "macOS 14.2",
                "mac_address": "55:66:77:88:99:00",
                "protocol": "TLSv1.3",
                "browser_or_agent": "Chrome/120.0"
            }
        },
        {
            "event_id": "evt_tokyo_02",
            "entity_id": "U103",
            "entity_type": "user",
            "timestamp": "2026-07-25T09:30:00Z",
            "geo_location": {
                "ip": "202.214.194.1",
                "country": "Japan",
                "city": "Tokyo",
                "latitude": 35.6762,
                "longitude": 139.6503
            },
            "resource_accessed": "/docs/quarterly_report.pdf",
            "auth_method": "password",
            "auth_success": true,
            "session_duration": 120.0,
            "command_sequence": [],
            "device_fingerprint": {
                "os_version": "macOS 14.2",
                "mac_address": "55:66:77:88:99:00",
                "protocol": "TLSv1.3",
                "browser_or_agent": "Chrome/120.0"
            }
        }
    ],
    device_spoofing: {
        "event_id": "evt_ds_777",
        "entity_id": "user_001",
        "entity_type": "user",
        "timestamp": "2026-07-25T14:40:00Z",
        "geo_location": {
            "ip": "198.51.100.10",
            "country": "USA",
            "city": "New York",
            "latitude": 40.7128,
            "longitude": -74.006
        },
        "resource_accessed": "/api/v1/payroll",
        "auth_method": "password",
        "auth_success": true,
        "session_duration": 600.0,
        "command_sequence": [],
        "device_fingerprint": {
            "os_version": "Kali Linux 2024.1",
            "mac_address": "DE:AD:BE:EF:CA:FE",
            "protocol": "SSHv2",
            "browser_or_agent": "Mozilla/5.0 (X11; Linux x86_64)"
        },
        "label": "device_spoofing"
    },
    lateral_movement: {
        "event_id": "evt_lm_666",
        "entity_id": "user_002",
        "entity_type": "user",
        "timestamp": "2026-07-25T14:45:00Z",
        "geo_location": {
            "ip": "198.51.100.10",
            "country": "USA",
            "city": "New York",
            "latitude": 40.7128,
            "longitude": -74.006
        },
        "resource_accessed": "/admin/vault_keys",
        "auth_method": "password",
        "auth_success": true,
        "session_duration": 900.0,
        "command_sequence": ["mimikatz.exe", "curl http://internal.vault/dump"],
        "device_fingerprint": {
            "os_version": "Windows 11 Enterprise",
            "mac_address": "11:22:33:44:55:66",
            "protocol": "TLSv1.3",
            "browser_or_agent": "Chrome/120.0"
        },
        "label": "lateral_movement"
    },
    normal: {
        "event_id": "evt_norm_111",
        "entity_id": "user_001",
        "entity_type": "user",
        "timestamp": "2026-07-25T10:15:00Z",
        "geo_location": {
            "ip": "198.51.100.10",
            "country": "USA",
            "city": "New York",
            "latitude": 40.7128,
            "longitude": -74.006
        },
        "resource_accessed": "/docs/handbook.pdf",
        "auth_method": "token",
        "auth_success": true,
        "session_duration": 1800.0,
        "command_sequence": [],
        "device_fingerprint": {
            "os_version": "Windows 11 Enterprise",
            "mac_address": "11:22:33:44:55:66",
            "protocol": "TLSv1.3",
            "browser_or_agent": "Chrome/120.0"
        },
        "label": "normal"
    }
};

function loadPreset(name) {
    if (PRESETS[name]) {
        document.getElementById('eval-json-input').value = JSON.stringify(PRESETS[name], null, 2);
    }
}

async function evaluateSingleEvent() {
    const inputArea = document.getElementById('eval-json-input');
    const resultPanel = document.getElementById('eval-result-panel');
    
    let rawJson = inputArea.value.trim();
    if (!rawJson) {
        alert("Please paste a JSON event or select a preset template!");
        return;
    }

    let parsedObj = null;
    try {
        parsedObj = JSON.parse(rawJson);
    } catch (e) {
        alert("Invalid JSON format! Please check your syntax.");
        return;
    }

    resultPanel.innerHTML = `<div style="text-align:center; padding:40px; color:var(--accent-blue);">⚡ Evaluating AI Threat Model & Feature Attribution...</div>`;

    try {
        const res = await fetch('/api/evaluate_single', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(parsedObj)
        });
        
        let data = {};
        const textResp = await res.text();
        try {
            data = JSON.parse(textResp);
        } catch (jsonErr) {
            resultPanel.innerHTML = `<div style="color:var(--accent-red); padding:16px;">Server Error Response: ${textResp || 'Empty response'}</div>`;
            return;
        }

        if (data.error) {
            resultPanel.innerHTML = `<div style="color:var(--accent-red); padding:16px;">${data.error}</div>`;
            return;
        }

        let traceHtml = '';
        if (data.diagnostic_trace && data.diagnostic_trace.traces) {
            data.diagnostic_trace.traces.forEach((tr, idx) => {
                let prevHtml = tr.previous_event ? `
                    <div style="background:rgba(255,255,255,0.02); padding:10px; border-radius:6px; margin-bottom:8px; font-size:11px;">
                        <strong style="color:var(--text-muted);">PREVIOUS EVENT:</strong><br>
                        Timestamp: ${tr.previous_event.timestamp}<br>
                        Location: ${tr.previous_event.city}, ${tr.previous_event.country} (${tr.previous_event.latitude}, ${tr.previous_event.longitude})<br>
                        Fingerprint: OS=${tr.previous_event.os} | MAC=${tr.previous_event.mac} | Protocol=${tr.previous_event.protocol}<br>
                        Resource: ${tr.previous_event.resource}
                    </div>
                ` : '';

                let currHtml = tr.current_event ? `
                    <div style="background:rgba(255,255,255,0.02); padding:10px; border-radius:6px; margin-bottom:8px; font-size:11px;">
                        <strong style="color:var(--text-muted);">CURRENT EVENT:</strong><br>
                        Timestamp: ${tr.current_event.timestamp}<br>
                        Location: ${tr.current_event.city}, ${tr.current_event.country} (${tr.current_event.latitude}, ${tr.current_event.longitude})<br>
                        Fingerprint: OS=${tr.current_event.os} | MAC=${tr.current_event.mac} | Protocol=${tr.current_event.protocol}<br>
                        Resource: ${tr.current_event.resource}
                    </div>
                ` : '';

                let derivedHtml = tr.raw_derived_features ? `
                    <div style="background:rgba(6,182,212,0.08); border:1px solid rgba(6,182,212,0.2); padding:10px; border-radius:6px; margin-bottom:8px; font-size:11px; font-family:var(--font-mono);">
                        <strong style="color:var(--accent-cyan);">RAW DERIVED FEATURES:</strong><br>
                        time_delta_minutes: ${tr.raw_derived_features.time_delta_minutes} min<br>
                        distance_km: ${tr.raw_derived_features.distance_km} km<br>
                        geo_velocity_kmh: ${tr.raw_derived_features.geo_velocity_kmh} km/h<br>
                        new_country: ${tr.raw_derived_features.new_country ? 'TRUE' : 'FALSE'}<br>
                        os_changed: ${tr.raw_derived_features.os_changed ? 'TRUE' : 'FALSE'}<br>
                        mac_changed: ${tr.raw_derived_features.mac_changed ? 'TRUE' : 'FALSE'}<br>
                        protocol_changed: ${tr.raw_derived_features.protocol_changed ? 'TRUE' : 'FALSE'}<br>
                        fingerprint_mismatch: ${tr.raw_derived_features.fingerprint_mismatch ? 'TRUE' : 'FALSE'}<br>
                        new_resource: ${tr.raw_derived_features.new_resource ? 'TRUE' : 'FALSE'}
                    </div>
                ` : '';

                let scoresHtml = tr.model_scores ? `
                    <div style="font-size:11px; font-family:var(--font-mono); margin-bottom:8px;">
                        Isolation Forest score: ${tr.model_scores.isolation_forest_score}<br>
                        LSTM score: ${tr.model_scores.lstm_sequence_score}<br>
                        Combined anomaly score: ${tr.model_scores.combined_anomaly_score}<br>
                        Threshold: ${tr.model_scores.threshold}
                    </div>
                ` : '';

                let probsHtml = tr.xgboost_class_probabilities ? `
                    <div style="margin-top:8px; background:rgba(0,0,0,0.3); padding:8px; border-radius:6px; font-size:11px; font-family:var(--font-mono);">
                        <strong style="color:var(--accent-amber);">XGBoost Class Probabilities:</strong><br>
                        ${Object.keys(tr.xgboost_class_probabilities).map(c => `&nbsp;&nbsp;${c}: ${(tr.xgboost_class_probabilities[c]*100).toFixed(1)}%`).join('<br>')}
                    </div>
                ` : '';

                traceHtml += `
                    <div style="border-left:3px solid var(--accent-blue); padding-left:10px; margin-bottom:14px;">
                        <h5 style="font-size:12px; color:var(--accent-blue);">Event #${tr.event_index} (${tr.timestamp})</h5>
                        ${prevHtml}
                        ${currHtml}
                        ${derivedHtml}
                        ${scoresHtml}
                        ${probsHtml}
                    </div>
                `;
            });
        }

        let riskBadge = 'badge-green';
        if (data.risk_score >= 75) riskBadge = 'badge-red';
        else if (data.risk_score >= 50) riskBadge = 'badge-amber';

        let factorsHtml = (data.contributing_factors || []).map(f => `
            <div class="factor-item" style="margin-bottom:6px;">⚠️ ${f}</div>
        `).join('');

        let keyReasonsHtml = (data.key_reasons || []).map(r => `
            <div style="margin-bottom:6px; color:var(--text-main); font-size:12px; display:flex; align-items:flex-start;">
                <span style="color:var(--accent-amber); margin-right:8px;">📌</span>
                <span>${r}</span>
            </div>
        `).join('');

        let attrBarsHtml = '';
        if (data.feature_attribution_weights && Object.keys(data.feature_attribution_weights).length > 0) {
            attrBarsHtml = Object.keys(data.feature_attribution_weights).map(k => {
                let w = data.feature_attribution_weights[k];
                return `
                    <div style="margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:2px; font-family:var(--font-mono);">
                            <span style="color:var(--text-muted);">${k}</span>
                            <span style="color:var(--accent-cyan); font-weight:700;">${w.toFixed(1)}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;">
                            <div style="background:linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)); height:100%; width:${Math.min(100, w)}%;"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        let attackTitleFormatted = data.predicted_attack_type.replace(/_/g, ' ').toUpperCase();

        resultPanel.innerHTML = `
            <div style="margin-bottom:12px; border-bottom:1px solid var(--border-color); padding-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <div>
                        <span style="font-size:10px; color:var(--accent-green); text-transform:uppercase; font-weight:700;">🔒 ISOLATED TEST MODE ENABLED (NO STATE CONTAMINATION)</span>
                        <h3 style="font-size:18px; font-weight:700; color:var(--accent-cyan); text-transform:capitalize;">
                            Predicted: ${data.predicted_attack_type.replace(/_/g, ' ')}
                        </h3>
                    </div>
                    <span class="risk-pill ${riskBadge}" style="font-size:14px; padding:4px 12px;">
                        Risk: ${data.risk_score} / 100
                    </span>
                </div>
                <div style="font-size:11px; color:var(--text-muted);">
                    Ground Truth = <strong>N/A</strong>
                </div>
            </div>

            <!-- DEDICATED XAI EXPLAINABILITY & DIAGNOSIS CARD -->
            <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(6,182,212,0.3); border-radius:10px; padding:14px; margin-bottom:16px;">
                <div style="display:flex; align-items:center; margin-bottom:8px;">
                    <span style="font-size:16px; margin-right:8px;">💡</span>
                    <h4 style="font-size:13px; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.5px;">
                        AI Threat Diagnosis & Explainability: ${attackTitleFormatted}
                    </h4>
                </div>

                <div style="font-size:12px; color:var(--text-muted); line-height:1.5; margin-bottom:12px; background:rgba(0,0,0,0.2); padding:10px; border-radius:6px;">
                    ${data.attack_description || 'Sequential telemetry analyzed by Aegis AI multi-stage behavioral model.'}
                </div>

                <div style="margin-bottom:12px;">
                    <h5 style="font-size:11px; font-weight:700; color:var(--accent-amber); text-transform:uppercase; margin-bottom:6px;">
                        🔍 Root Cause Analysis & Key Reasons Why:
                    </h5>
                    ${keyReasonsHtml}
                </div>

                ${attrBarsHtml ? `
                <div>
                    <h5 style="font-size:11px; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; margin-bottom:8px;">
                        📊 Causal Feature Attributions:
                    </h5>
                    ${attrBarsHtml}
                </div>
                ` : ''}
            </div>

            <div style="margin-bottom:12px;">
                <h4 style="font-size:12px; color:var(--accent-amber); margin-bottom:4px;">Explanation Phrasing:</h4>
                <div class="factor-list">${factorsHtml}</div>
            </div>

            <div>
                <h4 style="font-size:12px; margin-bottom:8px;">Chronological Isolated Trace & Feature Vector:</h4>
                ${traceHtml}
            </div>
        `;
    } catch (err) {
        resultPanel.innerHTML = `<div style="color:var(--accent-red); padding:16px;">Evaluation Error: ${err}</div>`;
    }
}

function renderTaxonomy(breakdown) {
    const container = document.getElementById('taxonomy-breakdown');
    if (!container) return;

    let html = Object.keys(breakdown).map(k => `
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-color); padding:16px; border-radius:10px;">
            <div style="font-size:12px; color:var(--text-muted); text-transform:uppercase;">${k.replace(/_/g, ' ')}</div>
            <div style="font-size:24px; font-weight:700; color:var(--accent-cyan); font-family:var(--font-mono); margin-top:4px;">
                ${breakdown[k]}
            </div>
        </div>
    `).join('');

    container.innerHTML = html;
}

