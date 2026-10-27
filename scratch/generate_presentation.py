"""
Script to generate an executive 12-slide presentation deck for Aegis AI
AI-Powered Behavioral Anomaly Detection & Threat Classification.
Creates Aegis_AI_Project_Presentation.pptx in 16:9 widescreen format.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return RGBColor(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))

# Global Palette - Dark Cyber SOC Theme
C_BG_DARK     = hex_to_rgb("0B1120")  # Deep space slate
C_BG_CARD     = hex_to_rgb("1E293B")  # Surface slate
C_BG_CARD_ALT = hex_to_rgb("152138")  # Darker card surface
C_BORDER      = hex_to_rgb("334155")  # Card border
C_BORDER_CYAN = hex_to_rgb("0284C7")  # Cyan accent border
C_CYAN        = hex_to_rgb("38BDF8")  # Highlight cyan
C_BLUE        = hex_to_rgb("3B82F6")  # Accent blue
C_GREEN       = hex_to_rgb("10B981")  # Emerald green
C_AMBER       = hex_to_rgb("F59E0B")  # Amber warning
C_RED         = hex_to_rgb("EF4444")  # Alert red
C_TEXT_WHITE  = hex_to_rgb("F8FAFC")  # Pure text
C_TEXT_MUTED  = hex_to_rgb("94A3B8")  # Muted secondary text
C_TEXT_DIM    = hex_to_rgb("64748B")  # Dim labels

def set_slide_background(slide, color=C_BG_DARK):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_badge=None, subtitle_text=None):
    top = 0.4
    if category_badge:
        # Category Badge Pill
        badge_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(top), Inches(2.8), Inches(0.32))
        badge_box.fill.solid()
        badge_box.fill.fore_color.rgb = hex_to_rgb("0369A1")
        badge_box.line.fill.background()
        tf = badge_box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.text = category_badge.upper()
        p.alignment = PP_ALIGN.CENTER
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb("E0F2FE")
        top += 0.42

    # Title
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(top), Inches(11.7), Inches(0.65))
    tf = t_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = "Calibri"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = "Calibri"
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = C_CYAN
        p2.space_before = Pt(3)

def add_card(slide, left, top, width, height, bg_color=C_BG_CARD, border_color=C_BORDER):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
    else:
        card.line.fill.background()
    return card

def add_stat_card(slide, left, top, width, height, stat_value, label, subtitle=None, accent_color=C_CYAN):
    add_card(slide, left, top, width, height, bg_color=C_BG_CARD, border_color=C_BORDER_CYAN)
    tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.15), Inches(width - 0.3), Inches(height - 0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    
    # Stat
    p = tf.paragraphs[0]
    p.text = str(stat_value)
    p.font.name = "Calibri"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = accent_color
    
    # Label
    p2 = tf.add_paragraph()
    p2.text = label
    p2.font.name = "Calibri"
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_WHITE
    p2.space_before = Pt(2)
    
    # Subtitle
    if subtitle:
        p3 = tf.add_paragraph()
        p3.text = subtitle
        p3.font.name = "Calibri"
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = C_TEXT_MUTED
        p3.space_before = Pt(2)

def add_bullet_run(paragraph, bold_prefix, text, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10)):
    r1 = paragraph.add_run()
    r1.text = bold_prefix
    r1.font.name = "Calibri"
    r1.font.bold = True
    r1.font.size = font_size
    r1.font.color.rgb = bold_color
    
    r2 = paragraph.add_run()
    r2.text = text
    r2.font.name = "Calibri"
    r2.font.size = font_size
    r2.font.color.rgb = text_color

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Cyber Dark Cover)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, hex_to_rgb("070D17"))

    # Glowing decorative strip on left
    strip = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.3), Inches(7.5))
    strip.fill.solid()
    strip.fill.fore_color.rgb = C_CYAN
    strip.line.fill.background()

    # Category Pill
    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.0), Inches(4.2), Inches(0.38))
    pill.fill.solid()
    pill.fill.fore_color.rgb = hex_to_rgb("0284C7")
    pill.line.fill.background()
    p = pill.text_frame.paragraphs[0]
    p.text = "AI / ML & CYBERSECURITY OPERATIONS"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = "Calibri"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb("FFFFFF")

    # Title Box
    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(10.8), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "AEGIS AI"
    p1.font.name = "Calibri"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = C_CYAN

    p2 = tf.add_paragraph()
    p2.text = "AI-Powered Behavioral Anomaly Detection & Threat Classification Engine"
    p2.font.name = "Calibri"
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_WHITE
    p2.space_before = Pt(6)

    p3 = tf.add_paragraph()
    p3.text = "Enterprise-Grade Decoupled Two-Stage Machine Learning for Near Real-Time Intrusion Discovery"
    p3.font.name = "Calibri"
    p3.font.size = Pt(13)
    p3.font.color.rgb = C_TEXT_MUTED
    p3.space_before = Pt(8)

    # 4 Quick Highlights Box at Bottom
    highlights = [
        ("95.36%", "System Accuracy", "Held-out test split"),
        ("0.9734", "Stage-1 ROC-AUC", "2.45% False Positive Rate"),
        ("0% Overlap", "Zero Data Leakage", "Strict entity grouping"),
        ("< 4.2 ms", "Streaming Latency", "Real-time REST API")
    ]
    card_w = 2.5
    for i, (val, lbl, sub) in enumerate(highlights):
        cx = 1.2 + i * (card_w + 0.3)
        add_stat_card(s1, cx, 4.2, card_w, 1.4, val, lbl, sub, accent_color=C_CYAN if i%2==0 else C_GREEN)

    # Author Metadata Card
    author_card = add_card(s1, 1.2, 5.9, 10.9, 0.9, bg_color=C_BG_CARD_ALT, border_color=C_BORDER)
    atb = s1.shapes.add_textbox(Inches(1.4), Inches(5.95), Inches(10.5), Inches(0.8))
    atf = atb.text_frame
    atf.word_wrap = True
    ap1 = atf.paragraphs[0]
    ap1.text = "Presenter: Venkata Abhinav Kumar Gupta Gunda  |  Registration ID: 23BCE1037"
    ap1.font.name = "Calibri"
    ap1.font.size = Pt(12)
    ap1.font.bold = True
    ap1.font.color.rgb = C_TEXT_WHITE

    ap2 = atf.add_paragraph()
    ap2.text = "GitHub: github.com/abhinav260506/Behavioral-Anomaly-Detection-  |  Stack: Python, PyTorch, XGBoost, Scikit-Learn"
    ap2.font.name = "Calibri"
    ap2.font.size = Pt(10.5)
    ap2.font.color.rgb = C_CYAN

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & INDUSTRY CHALLENGES
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Problem Statement & Industry Bottlenecks", "THE CYBERSECURITY CHALLENGE", "Why traditional rule-based and signature-matching SIEM systems fail in modern enterprises")

    col_w = 3.65
    c_y = 1.6
    c_h = 5.3

    # Col 1: Legacy Failure
    add_card(s2, 0.8, c_y, col_w, c_h, border_color=C_RED)
    tb1 = s2.shapes.add_textbox(Inches(0.95), Inches(c_y + 0.2), Inches(col_w - 0.3), Inches(c_h - 0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Legacy SIEM Shortcomings"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_RED

    points_1 = [
        ("Static Rule Bypass: ", "Attackers bypass fixed thresholds (e.g. 'fail > 5 logins') using low-and-slow rate-limited probes."),
        ("Blind to Zero-Days: ", "Hash/signature matching is completely blind to novel exploits, stolen valid tokens, and living-off-the-land commands."),
        ("No Entity Context: ", "Treats access logs as isolated rows without tracking user identity baselines, device fingerprints, or resource graphs.")
    ]
    for b_title, b_desc in points_1:
        bp = tf1.add_paragraph()
        bp.space_before = Pt(12)
        add_bullet_run(bp, b_title, b_desc, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10.5))

    # Col 2: Machine Learning Hurdles
    add_card(s2, 4.8, c_y, col_w, c_h, border_color=C_AMBER)
    tb2 = s2.shapes.add_textbox(Inches(4.95), Inches(c_y + 0.2), Inches(col_w - 0.3), Inches(c_h - 0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Machine Learning Hurdles"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_AMBER

    points_2 = [
        ("Extreme Class Imbalance: ", "~98% of enterprise traffic is normal, while intrusions comprise < 2%. Models easily collapse into majority-class predictors."),
        ("Cold-Start Dilemma: ", "Newly onboarded employees, service accounts, and edge IoT devices lack historical baseline data for statistical profiling."),
        ("Concept Drift: ", "Normal employee habits evolve naturally (shift changes, travel, role rotations), triggering false risk spikes.")
    ]
    for b_title, b_desc in points_2:
        bp = tf2.add_paragraph()
        bp.space_before = Pt(12)
        add_bullet_run(bp, b_title, b_desc, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10.5))

    # Col 3: SOC Analyst Impact
    add_card(s2, 8.8, c_y, col_w, c_h, border_color=C_CYAN)
    tb3 = s2.shapes.add_textbox(Inches(8.95), Inches(c_y + 0.2), Inches(col_w - 0.3), Inches(c_h - 0.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "The SOC Operational Crisis"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    points_3 = [
        ("Catastrophic Alert Fatigue: ", "High false positive rates flood analyst consoles with thousands of false alarms, drowning out genuine breaches."),
        ("The Black-Box Gap: ", "Standard neural networks output scalar risk scores without explaining WHY an event was flagged, stalling rapid triage."),
        ("Aegis AI Mission: ", "Deliver an automated, explainable, two-stage system with < 2.5% FPR and instant root-cause diagnostics.")
    ]
    for b_title, b_desc in points_3:
        bp = tf3.add_paragraph()
        bp.space_before = Pt(12)
        add_bullet_run(bp, b_title, b_desc, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10.5))

    # =========================================================================
    # SLIDE 3: AIM & CORE OBJECTIVES
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Project Aim & Specific Technical Objectives", "PROJECT MISSION", "Bridging unsupervised baseline modeling with explainable supervised threat taxonomy")

    # Main Aim Card Across Top
    add_card(s3, 0.8, 1.5, 11.7, 1.2, bg_color=C_BG_CARD_ALT, border_color=C_CYAN)
    atb = s3.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(11.3), Inches(1.1))
    atf = atb.text_frame
    atf.word_wrap = True
    p = atf.paragraphs[0]
    p.text = "PROJECT AIM:"
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    p2 = atf.add_paragraph()
    p2.text = "To research, engineer, and validate Aegis AI: a decoupled two-stage machine learning pipeline that models normal access baselines without ground-truth label leakage, achieves < 2.5% FPR on extreme class imbalance (~98% normal), classifies intrusions into 7 distinct attack categories, and generates Explainable AI root-cause diagnostics for SOC operations."
    p2.font.name = "Calibri"
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT_WHITE
    p2.space_before = Pt(3)

    # 4 Pillar Cards
    pillars = [
        ("1. Sequential Synthesis", "Synthetic generator producing 28,257 events across 165 entities (Users, Services, Edge IoT) with realistic temporal distributions and 8 attack behaviors.", C_BLUE),
        ("2. Decoupled 2-Stage ML", "Stage 1 (Unsupervised Isolation Forest + PyTorch LSTM Autoencoder) + Stage 2 (Supervised XGBoost Classifier trained with training-only SMOTE oversampling).", C_CYAN),
        ("3. Adaptive Baseline Profiler", "Cold-Start Peer Group Profiler for new entities (N < 10 logs) and Exponentially Weighted Moving Average (EWMA, alpha=0.05) to track legitimate habit drift.", C_GREEN),
        ("4. Explainable AI & SOC UI", "Automated root-cause translation engine providing percentage feature attribution bars and native REST API dashboard at http://localhost:5000.", C_AMBER)
    ]
    pw = 2.75
    for i, (p_title, p_desc, p_col) in enumerate(pillars):
        px = 0.8 + i * (pw + 0.23)
        add_card(s3, px, 2.9, pw, 4.0, border_color=p_col)
        ptb = s3.shapes.add_textbox(Inches(px + 0.15), Inches(3.05), Inches(pw - 0.3), Inches(3.6))
        ptf = ptb.text_frame
        ptf.word_wrap = True
        p = ptf.paragraphs[0]
        p.text = p_title
        p.font.name = "Calibri"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = p_col

        p2 = ptf.add_paragraph()
        p2.text = p_desc
        p2.font.name = "Calibri"
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = C_TEXT_MUTED
        p2.space_before = Pt(8)

    # =========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE & PIPELINE FLOW
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "System Architecture: Decoupled Two-Stage ML", "TECHNICAL ARCHITECTURE", "Separating unsupervised anomaly detection from supervised threat taxonomy classification")

    # Left: Architecture Flow Diagram Image
    add_card(s4, 0.8, 1.5, 6.2, 5.4, border_color=C_BORDER)
    if os.path.exists("Architechture diagram.png"):
        s4.shapes.add_picture("Architechture diagram.png", Inches(0.9), Inches(1.6), Inches(6.0), Inches(5.1))
    else:
        tb = s4.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.8), Inches(4.0))
        tb.text_frame.text = "Architecture Diagram: Stage 1 (Unsupervised) -> Threshold T=32.0 -> Stage 2 (Supervised XGBoost)"

    # Right: Pipeline Breakdown Cards
    right_x = 7.2
    right_w = 5.3

    arch_steps = [
        ("Incoming Raw Telemetry: ", "Structured JSON events (timestamp, geo-coordinates, IP, resource path, credentials, device fingerprint).", C_TEXT_WHITE),
        ("45-Dimensional Causal Features: ", "Extracts temporal cycles, continuous geo-velocity (km/h), rolling failure bursts, fingerprint shifts, and resource graph novelty.", C_CYAN),
        ("Stage 1: Unsupervised Ensemble: ", "Isolation Forest + PyTorch LSTM Autoencoder + Geo-Velocity scaling engine score baseline deviation into a 0-100 composite risk score.", C_BLUE),
        ("Calibrated Threshold Gate (T = 32.0): ", "Risk < 32.0 -> Benign (EWMA baseline updated). Risk >= 32.0 -> Flagged as Anomaly and dispatched to Stage 2.", C_AMBER),
        ("Stage 2: Supervised Attack Classifier: ", "XGBoost 150-tree model assigns the exact attack category and generates plain-language XAI root causes for the SOC alert console.", C_GREEN)
    ]

    sy = 1.5
    sh = 0.98
    for st_title, st_desc, st_col in arch_steps:
        add_card(s4, right_x, sy, right_w, sh, bg_color=C_BG_CARD, border_color=C_BORDER)
        stb = s4.shapes.add_textbox(Inches(right_x + 0.15), Inches(sy + 0.08), Inches(right_w - 0.3), Inches(sh - 0.15))
        stf = stb.text_frame
        stf.word_wrap = True
        p = stf.paragraphs[0]
        add_bullet_run(p, st_title, st_desc, bold_color=st_col, text_color=C_TEXT_MUTED, font_size=Pt(10))
        sy += sh + 0.12

    # =========================================================================
    # SLIDE 5: 45-DIMENSIONAL CAUSAL FEATURE ENGINEERING
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Causal Feature Engineering: 45 Specialized Metrics", "FEATURE EXTRACTION ENGINE", "Comprehensive domain-specific metrics capturing temporal, spatial, graph, and device telemetry")

    feat_cards = [
        ("1. Temporal Dynamics (4)", [
            "hour_sin / hour_cos: Cyclical 24-hour trigonometric encoding",
            "is_weekend: Binary indicator for non-business days",
            "is_off_hours: Access outside habitual 08:00 - 18:00 window"
        ], C_BLUE),
        ("2. Geo-Velocity & Spatial (4)", [
            "geo_velocity_kmh: Haversine speed between consecutive logins",
            "time_delta_minutes: Elapsed minutes since previous event",
            "distance_km: Physical spherical distance traversed",
            "impossible_travel_flag: Speeds physically exceeding 600 km/h"
        ], C_CYAN),
        ("3. Auth & Session (5)", [
            "rolling_failures_5m / 1h: Short & medium burst failure counters",
            "auth_failed: Binary authentication rejection flag",
            "ip_diversity_1h: Unique IP count within sliding 1-hour window",
            "session_duration: Logged session connection duration (seconds)"
        ], C_AMBER),
        ("4. Cryptographic Fingerprint (7)", [
            "os_changed / mac_changed: Hardware and OS profile shifts",
            "protocol_changed: TLSv1.3 vs SSHv2 communication alterations",
            "fingerprint_change_count_24h: Volatility frequency index",
            "time_since_last_change: Recency of hardware/browser alteration"
        ], C_RED),
        ("5. Graph & Volumetric Drift (25)", [
            "new_resource / resource_novelty: Unseen endpoints & graph paths",
            "unique_resources_10m / 1h: Breadth expansion over time",
            "restricted_resource / command_sequence_anomaly: Mimikatz flags",
            "download_volume_1h / 24h / 7d: Cumulative exfiltration tracking",
            "privilege_expansion_rate & behavior_drift_score: Role drift"
        ], C_GREEN)
    ]

    card_positions = [
        (0.8, 1.5, 3.7, 2.6),
        (4.8, 1.5, 3.7, 2.6),
        (8.8, 1.5, 3.7, 2.6),
        (0.8, 4.3, 5.7, 2.6),
        (6.8, 4.3, 5.7, 2.6)
    ]

    for (c_title, c_items, c_col), (lx, ty, w, h) in zip(feat_cards, card_positions):
        add_card(s5, lx, ty, w, h, border_color=c_col)
        tb = s5.shapes.add_textbox(Inches(lx + 0.15), Inches(ty + 0.15), Inches(w - 0.3), Inches(h - 0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = c_title
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = c_col

        for item in c_items:
            bp = tf.add_paragraph()
            bp.text = "• " + item
            bp.font.name = "Calibri"
            bp.font.size = Pt(9.5)
            bp.font.color.rgb = C_TEXT_MUTED
            bp.space_before = Pt(3)

    # =========================================================================
    # SLIDE 6: STAGE 1 — UNSUPERVISED ANOMALY DETECTION
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Stage 1: Unsupervised Behavioral Anomaly Detection", "STAGE-1 DETECTION ENGINE", "Zero-label ensemble modeling baseline normality and capturing physical and sequence deviations")

    s1_components = [
        ("Isolation Forest (100 Trees)", 
         "Fits on normal baseline training telemetry (contamination=0.02) without attack labels. Isolates anomalous vectors through random recursive partitioning, identifying sparse regions via shorter path lengths.",
         "Path Length Anomaly Score", C_BLUE),
        ("PyTorch LSTM Autoencoder", 
         "Deep sequence autoencoder (45 -> 32 -> 16 -> 32 -> 45) trained on normal user event sequences using Adam optimizer (lr=0.005). Intrusions exhibit high reconstruction Mean Squared Error (MSE).",
         "Sequence Reconstruction Loss", C_CYAN),
        ("Continuous Geo-Velocity (S_geo)", 
         "Calculates spherical Haversine distance. Speeds < 600 km/h incur 0 penalty. Speeds > 600 km/h scale continuously via S_geo = min(1.0, 0.35 + (velocity - 600)/3000), eliminating binary edge-case clipping.",
         "Physical Physics Constraints", C_AMBER),
        ("Cold-Start & Concept Drift Profiler", 
         "Blends new entity scores with peer-group population baselines using w_ind = min(1, N/10). EWMA exponential smoothing (alpha=0.05) updates normal profiles dynamically as benign habits evolve.",
         "Adaptive Dynamic Profiling", C_GREEN)
    ]

    card_w = 5.7
    card_h = 2.45
    positions = [
        (0.8, 1.5),
        (6.8, 1.5),
        (0.8, 4.3),
        (6.8, 4.3)
    ]

    for (c_title, c_desc, c_tag, c_col), (lx, ty) in zip(s1_components, positions):
        add_card(s6, lx, ty, card_w, card_h, border_color=c_col)
        tb = s6.shapes.add_textbox(Inches(lx + 0.2), Inches(ty + 0.15), Inches(card_w - 0.4), Inches(card_h - 0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        
        # Tag pill
        p = tf.paragraphs[0]
        p.text = c_tag.upper()
        p.font.name = "Calibri"
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = c_col
        
        p2 = tf.add_paragraph()
        p2.text = c_title
        p2.font.name = "Calibri"
        p2.font.size = Pt(14)
        p2.font.bold = True
        p2.font.color.rgb = C_TEXT_WHITE
        p2.space_before = Pt(2)

        p3 = tf.add_paragraph()
        p3.text = c_desc
        p3.font.name = "Calibri"
        p3.font.size = Pt(10)
        p3.font.color.rgb = C_TEXT_MUTED
        p3.space_before = Pt(4)

    # =========================================================================
    # SLIDE 7: STAGE 2 — SUPERVISED THREAT TAXONOMY CLASSIFICATION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Stage 2: Supervised Threat Taxonomy Classification", "STAGE-2 CLASSIFICATION ENGINE", "Categorizing confirmed anomalies into 7 MITRE-aligned cyber attack vectors using XGBoost")

    # Left: Model Description & SMOTE Integrity
    add_card(s7, 0.8, 1.5, 5.6, 5.4, border_color=C_CYAN)
    ltb = s7.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    ltf = ltb.text_frame
    ltf.word_wrap = True
    
    p = ltf.paragraphs[0]
    p.text = "XGBoost Classifier Architecture"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    xgb_pts = [
        ("150 Gradient Boosted Trees: ", "Max depth = 6, learning rate = 0.1, objective = multi:softprob across 7 attack classes."),
        ("Strict Training-Only SMOTE: ", "Synthetic Minority Over-sampling is applied ONLY to Stage-2 training attack vectors (k_neighbors=3). Zero synthetic samples ever enter validation or test splits."),
        ("Zero Execution on Normal Traffic: ", "Filters out ~98% of traffic at Stage 1, invoking XGBoost only on flagged events (~2%), achieving high computational efficiency."),
        ("Softmax Class Probabilities: ", "Outputs calibrated probability distributions across all 7 categories for clear multi-vector risk profiling.")
    ]
    for pt_t, pt_d in xgb_pts:
        bp = ltf.add_paragraph()
        bp.space_before = Pt(10)
        add_bullet_run(bp, pt_t, pt_d, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10))

    # Right: The 7 Modeled Attack Categories
    add_card(s7, 6.7, 1.5, 5.8, 5.4, border_color=C_GREEN)
    rtb = s7.shapes.add_textbox(Inches(6.9), Inches(1.7), Inches(5.4), Inches(5.0))
    rtf = rtb.text_frame
    rtf.word_wrap = True

    p = rtf.paragraphs[0]
    p.text = "The 7 MITRE-Aligned Attack Vectors"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_GREEN

    attacks = [
        ("Brute Force: ", "High-frequency authentication failures targeting single login endpoints."),
        ("Credential Stuffing: ", "Distributed multi-account password guessing from shared NAT/proxy pools."),
        ("Lateral Movement: ", "Traversal into internal vaults (/admin/vault_keys) with credential dump tools."),
        ("Device Spoofing: ", "Sudden changes in MAC address, OS version (Windows -> Kali), or TLS protocol."),
        ("Low-and-Slow Exfil: ", "Small off-hours transfers accumulating over extended multi-day rolling windows."),
        ("Impossible Travel: ", "Logins from distant cities (Chennai -> Tokyo) within impossible time deltas."),
        ("Insider Drift: ", "Legitimate entity gradually expanding resource footprint across unauthorized departments.")
    ]
    for at_t, at_d in attacks:
        bp = rtf.add_paragraph()
        bp.space_before = Pt(5)
        add_bullet_run(bp, at_t, at_d, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(9.5))

    # =========================================================================
    # SLIDE 8: DATASET ENGINEERING & ZERO-LEAKAGE SPLIT PROTOCOL
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Dataset Engineering & Zero-Leakage Splitting", "DATASET RIGOR", "28,257 access events across 165 entities with strict entity-grouped held-out partitioning")

    # 3 Stat Cards for Splits
    splits = [
        ("70% Train", "115 Entities (19,828 logs)", "Baseline profile fitting, scaler normalization, and Stage-1/Stage-2 model fitting.", C_BLUE),
        ("15% Validation", "24 Entities (3,948 logs)", "Held out strictly for calibrating the Stage-1 risk cutoff threshold (T = 32.0).", C_AMBER),
        ("15% Held-Out Test", "26 Entities (4,481 logs)", "Untouched during all training. Evaluates pure generalization on brand-new entities.", C_GREEN)
    ]
    sw = 3.7
    for i, (s_pct, s_ent, s_desc, s_col) in enumerate(splits):
        sx = 0.8 + i * (sw + 0.3)
        add_card(s8, sx, 1.5, sw, 2.0, border_color=s_col)
        tb = s8.shapes.add_textbox(Inches(sx + 0.15), Inches(1.6), Inches(sw - 0.3), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = s_pct
        p.font.name = "Calibri"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = s_col

        p2 = tf.add_paragraph()
        p2.text = s_ent
        p2.font.name = "Calibri"
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = C_TEXT_WHITE
        p2.space_before = Pt(2)

        p3 = tf.add_paragraph()
        p3.text = s_desc
        p3.font.name = "Calibri"
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = C_TEXT_MUTED
        p3.space_before = Pt(3)

    # Bottom Half: Data Integrity & Entity Grouping Callout
    add_card(s8, 0.8, 3.8, 11.7, 3.1, bg_color=C_BG_CARD_ALT, border_color=C_BORDER)
    btb = s8.shapes.add_textbox(Inches(1.0), Inches(3.95), Inches(11.3), Inches(2.8))
    btf = btb.text_frame
    btf.word_wrap = True

    p = btf.paragraphs[0]
    p.text = "Why Entity-Grouped Partitioning Eliminates Temporal Data Leakage:"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    reasons = [
        ("The Flaw of Random Splitting: ", "Standard random row-level k-fold cross validation leaks past user behavior into future evaluation, artificially inflating accuracy scores while failing in live enterprise deployments."),
        ("Strict 0% Entity Overlap: ", "In Aegis AI, the 26 test entities appear ZERO times in training or validation data. Every test entity represents a completely unobserved individual, service account, or IoT device."),
        ("Realistic Extreme Imbalance Maintained: ", "The natural ~98% normal distribution is preserved across all validation and test sets. No synthetic samples are ever inserted into the held-out evaluation stream.")
    ]
    for r_t, r_d in reasons:
        bp = btf.add_paragraph()
        bp.space_before = Pt(6)
        add_bullet_run(bp, r_t, r_d, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10))

    # =========================================================================
    # SLIDE 9: EMPIRICAL BENCHMARK RESULTS
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Empirical Evaluation Results: Untouched Test Split", "VERIFIED BENCHMARKS", "Evaluated on 4,481 logs across 26 entities with zero model fitting executed during testing")

    # 4 Stat Cards
    metrics = [
        ("95.36%", "End-to-End Accuracy", "On untouched held-out test split", C_CYAN),
        ("0.9734", "Stage-1 ROC-AUC", "PR-AUC = 0.7266", C_GREEN),
        ("2.45%", "False Positive Rate", "Comfortably below SOC 5% limit", C_BLUE),
        ("< 4.2 ms", "Streaming Latency", "Sub-second real-time scoring", C_AMBER)
    ]
    mw = 2.75
    for i, (m_val, m_lbl, m_sub, m_col) in enumerate(metrics):
        mx = 0.8 + i * (mw + 0.23)
        add_stat_card(s9, mx, 1.5, mw, 1.4, m_val, m_lbl, m_sub, accent_color=m_col)

    # Bottom Half: Per-Class Performance Table
    table_shape = s9.shapes.add_table(9, 5, Inches(0.8), Inches(3.2), Inches(11.7), Inches(3.7))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(3.3)
    tbl.columns[1].width = Inches(2.0)
    tbl.columns[2].width = Inches(2.1)
    tbl.columns[3].width = Inches(2.1)
    tbl.columns[4].width = Inches(2.2)

    headers = ["Threat Category", "Test Support", "Precision", "Recall", "F1-Score"]
    for c_idx, h_text in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = hex_to_rgb("0F172A")
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = hex_to_rgb("FFFFFF")
        p.alignment = PP_ALIGN.CENTER

    table_data = [
        ("Normal Baseline", "4,122", "97.43%", "97.55%", "97.49%"),
        ("Lateral Movement", "96", "89.72%", "100.00%", "94.58%"),
        ("Brute Force", "45", "100.00%", "82.22%", "90.24%"),
        ("Credential Stuffing", "61", "100.00%", "78.69%", "88.07%"),
        ("Device Spoofing", "21", "58.33%", "66.67%", "62.22%"),
        ("Insider Drift", "70", "43.33%", "55.71%", "48.75%"),
        ("Low-and-Slow Exfiltration", "42", "34.21%", "30.95%", "32.50%"),
        ("Impossible Travel", "24", "50.00%", "20.83%", "29.41%")
    ]
    for r_idx, row_vals in enumerate(table_data, start=1):
        bg_col = hex_to_rgb("152138") if r_idx % 2 == 1 else hex_to_rgb("1E293B")
        for c_idx, val in enumerate(row_vals):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_col
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            p.font.color.rgb = C_TEXT_WHITE if c_idx == 0 or c_idx == 4 else C_TEXT_MUTED
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
            if c_idx == 4:
                p.font.bold = True
                p.font.color.rgb = C_CYAN

    # =========================================================================
    # SLIDE 10: EXPLAINABLE AI (XAI) & SOC OPERATIONS CENTER
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Explainable AI (XAI) & SOC Operations Center", "SOC ANALYST USABILITY", "Automated plain-language root-cause diagnostics and lightweight REST API dashboard")

    # Left: XAI Diagnostic Card
    add_card(s10, 0.8, 1.5, 5.6, 5.4, border_color=C_CYAN)
    xtb = s10.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    xtf = xtb.text_frame
    xtf.word_wrap = True

    p = xtf.paragraphs[0]
    p.text = "Automated XAI Root Cause Engine"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    xai_points = [
        ("Transparent Attribution: ", "Translates mathematical model vectors into exact quantitative evidence (e.g. 'Traveling speed 12,821 km/h between logins (threshold: 600 km/h)')."),
        ("Percentage Weight Bars: ", "Ranks contributing features by normalized deviation importance so analysts know whether an alert is driven by IP diversity, auth failure bursts, or command anomalies."),
        ("No Black-Box Stalling: ", "Empowers Tier-1 analysts to immediately triage, confirm, or dismiss alerts without requiring data science escalation."),
        ("Analyst Feedback Loop: ", "One-click 'Confirm Threat' or 'False Positive' buttons trigger POST /api/feedback to dynamically adjust EWMA drift boundaries.")
    ]
    for xp_t, xp_d in xai_points:
        bp = xtf.add_paragraph()
        bp.space_before = Pt(8)
        add_bullet_run(bp, xp_t, xp_d, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10))

    # Right: SOC Dashboard Modules
    add_card(s10, 6.7, 1.5, 5.8, 5.4, border_color=C_GREEN)
    dtb = s10.shapes.add_textbox(Inches(6.9), Inches(1.7), Inches(5.4), Inches(5.0))
    dtf = dtb.text_frame
    dtf.word_wrap = True

    p = dtf.paragraphs[0]
    p.text = "Interactive SOC Web Application"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_GREEN

    p_sub = dtf.add_paragraph()
    p_sub.text = "Active on http://localhost:5000 (Python native HTTP server)"
    p_sub.font.name = "Calibri"
    p_sub.font.size = Pt(10)
    p_sub.font.color.rgb = C_TEXT_MUTED
    p_sub.space_before = Pt(2)

    dash_modules = [
        ("Ranked Alert Queue: ", "Live event stream ranked by composite risk score (0-100) with severity badges (Critical, High, Medium, Low)."),
        ("Interactive Attack Simulator: ", "Inject live attack sequences (Brute Force, Credential Stuffing, Impossible Travel) into the telemetry stream in real time."),
        ("Single Event Evaluator: ", "Test arbitrary JSON access logs or pre-built attack presets in isolated mode without contaminating global baselines."),
        ("Entity Behavioral Profiler: ", "Inspect historical access timelines, device fingerprints, and peer-group baseline comparisons."),
        ("Model Verification Tab: ", "Live SHA256 checksums and confusion matrix analytics.")
    ]
    for dm_t, dm_d in dash_modules:
        bp = dtf.add_paragraph()
        bp.space_before = Pt(6)
        add_bullet_run(bp, dm_t, dm_d, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(9.5))

    # =========================================================================
    # SLIDE 11: FEASIBILITY, VIABILITY & FUTURE ROADMAP
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Feasibility, Viability & Future Roadmap", "DEPLOYMENT READINESS", "Enterprise scalability, SIEM compatibility, and future research expansions")

    col_w = 3.65
    c_y = 1.5
    c_h = 5.4

    # Card 1: Feasibility
    add_card(s11, 0.8, c_y, col_w, c_h, border_color=C_BLUE)
    tb1 = s11.shapes.add_textbox(Inches(0.95), Inches(c_y + 0.2), Inches(col_w - 0.3), Inches(c_h - 0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "Operational Feasibility"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    feas_pts = [
        ("Lightweight Compute: ", "Decoupled architecture filters ~98% of traffic using fast tree/matrix operations, invoking heavy XGBoost only on ~2%."),
        ("Real-Time Latency: ", "< 4.2 ms / event enables seamless streaming ingestion into production SIEMs (Splunk, Microsoft Sentinel)."),
        ("Zero Heavy Frameworks: ", "Runs out of the box using native Python HTTP server without requiring complex Docker/K8s clusters.")
    ]
    for b_title, b_desc in feas_pts:
        bp = tf1.add_paragraph()
        bp.space_before = Pt(10)
        add_bullet_run(bp, b_title, b_desc, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10))

    # Card 2: Viability
    add_card(s11, 4.8, c_y, col_w, c_h, border_color=C_GREEN)
    tb2 = s11.shapes.add_textbox(Inches(4.95), Inches(c_y + 0.2), Inches(col_w - 0.3), Inches(c_h - 0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "Commercial Viability"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_GREEN

    viab_pts = [
        ("Alert Budget Adherence: ", "2.45% False Positive Rate preserves analyst sanity and strictly respects enterprise alert budgets."),
        ("No Costly Labeling: ", "Stage 1 operates completely unsupervised, eliminating the need for expensive security analyst manual data tagging."),
        ("Model Tamper-Proofing: ", "Frozen serialized artifacts under models/ with SHA256 checksums ensure regulatory compliance and auditability.")
    ]
    for b_title, b_desc in viab_pts:
        bp = tf2.add_paragraph()
        bp.space_before = Pt(10)
        add_bullet_run(bp, b_title, b_desc, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10))

    # Card 3: Future Roadmap
    add_card(s11, 8.8, c_y, col_w, c_h, border_color=C_CYAN)
    tb3 = s11.shapes.add_textbox(Inches(8.95), Inches(c_y + 0.2), Inches(col_w - 0.3), Inches(c_h - 0.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "Future Technical Roadmap"
    p.font.name = "Calibri"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    road_pts = [
        ("Graph Neural Networks: ", "Integrate GraphSAGE / Relational GCNs to track complex multi-hop Active Directory domain traversal."),
        ("Local LLM Playbooks: ", "Hook XAI attribution vectors into local Llama-3-8B to auto-generate remediation playbooks and firewall rules."),
        ("Edge Device ONNX: ", "Quantize PyTorch LSTM Autoencoder to INT8 for deployment on low-power OT/IoT hardware firewalls.")
    ]
    for b_title, b_desc in road_pts:
        bp = tf3.add_paragraph()
        bp.space_before = Pt(10)
        add_bullet_run(bp, b_title, b_desc, bold_color=C_TEXT_WHITE, text_color=C_TEXT_MUTED, font_size=Pt(10))

    # =========================================================================
    # SLIDE 12: CONCLUSION & PROJECT SUMMARY
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, hex_to_rgb("070D17"))

    # Left Strip
    strip = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.3), Inches(7.5))
    strip.fill.solid()
    strip.fill.fore_color.rgb = C_GREEN
    strip.line.fill.background()

    # Title
    t_box = s12.shapes.add_textbox(Inches(1.2), Inches(1.0), Inches(10.8), Inches(1.5))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "CONCLUSION & SUMMARY"
    p1.font.name = "Calibri"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = C_TEXT_WHITE

    p2 = tf.add_paragraph()
    p2.text = "Aegis AI delivers an enterprise-ready behavioral intrusion detection engine solving extreme class imbalance, cold start, and explainability."
    p2.font.name = "Calibri"
    p2.font.size = Pt(14)
    p2.font.color.rgb = C_CYAN
    p2.space_before = Pt(6)

    # 3 Summary Key Highlight Cards
    summary_cards = [
        ("95.36% Accuracy", "Zero-Leakage Benchmark", "Proven on 26 unobserved test entities with 0% entity overlap.", C_CYAN),
        ("2.45% False Positives", "ROC-AUC = 0.9734", "Stops alert fatigue and respects realistic SOC triage constraints.", C_GREEN),
        ("Production Deployed", "Live Operations Center", "Active REST API, test engine, and interactive dashboard at localhost:5000.", C_BLUE)
    ]
    cw = 3.4
    for i, (sc_val, sc_lbl, sc_sub, sc_col) in enumerate(summary_cards):
        cx = 1.2 + i * (cw + 0.35)
        add_stat_card(s12, cx, 2.7, cw, 1.8, sc_val, sc_lbl, sc_sub, accent_color=sc_col)

    # Bottom Contact & Repository Card
    bot_card = add_card(s12, 1.2, 4.8, 10.9, 1.8, bg_color=C_BG_CARD, border_color=C_BORDER)
    btb = s12.shapes.add_textbox(Inches(1.4), Inches(4.95), Inches(10.5), Inches(1.5))
    btf = btb.text_frame
    btf.word_wrap = True

    p = btf.paragraphs[0]
    p.text = "Project Resources & Repository Links:"
    p.font.name = "Calibri"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE

    links = [
        ("GitHub Repository: ", "https://github.com/abhinav260506/Behavioral-Anomaly-Detection-"),
        ("Live Web Dashboard: ", "http://localhost:5000 (Launch via run.bat)"),
        ("Lead Developer: ", "Venkata Abhinav Kumar Gupta Gunda (23BCE1037 | gunda.abhinav26@gmail.com)")
    ]
    for l_t, l_v in links:
        bp = btf.add_paragraph()
        bp.space_before = Pt(4)
        add_bullet_run(bp, l_t, l_v, bold_color=C_CYAN, text_color=C_TEXT_WHITE, font_size=Pt(10.5))

    # Save presentation
    output_filename = "Aegis_AI_Project_Presentation.pptx"
    prs.save(output_filename)
    print(f"[+] Successfully generated 16:9 presentation: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    build_presentation()
