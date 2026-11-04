"""Calibration constants for the fake Campus Cham dataset.

Source keys [S1]..[S10] refer to docs/calibration.md. "ASSUMPTION" marks values with no
published figure; they are the generator's weakest points and are listed in the report.
"""

# [S2] THD Zahlen-Daten-Fakten WS 2025/26: 913 students at Campus Cham.
CAMPUS_CHAM_STUDENTS = 913
CURRENT_YEAR = 2026  # winter term 2026/27

# [S1] programmes at Campus Cham. Shares per programme are not published: ASSUMPTION.
PROGRAMMES = {
    # code: (name, degree, language, share, standard semesters)
    "MDP": ("Mechatronik & Projektmanagement für Digitale Produktion", "Bachelor", "de", 0.25, 7),
    "MMC": ("Mechatronic and Cyber-Physical Systems", "Master", "en", 0.20, 3),
    "MSS": ("Artificial Intelligence for Smart Sensors and Actuators", "Master", "en", 0.20, 3),
    "MDM": ("Applied AI for Digital Production Management", "Master", "en", 0.20, 3),
    "MRO": ("Robotics", "Master", "en", 0.15, 3),
}

# International share. THD overall 51.3 % [S2]; Cham not published. ASSUMPTION: German-taught
# Bachelor mostly domestic, English-taught Masters mostly international (overall ~68 %).
INTERNATIONAL_SHARE = {"Bachelor": 0.15, "Master": 0.85}

# Countries of origin of international students in Germany, WS 2024/25 [S3] (national proxy).
# Head of the distribution is sourced; the long tail (remaining 56 %) is an ASSUMPTION spread
# over countries plausible for an English-taught engineering campus near the Czech border.
ORIGIN_SOURCED = {
    "India": 0.146, "China": 0.097, "Türkiye": 0.052, "Iran": 0.043,
    "Austria": 0.039, "Pakistan": 0.031, "Syria": 0.029,
}
ORIGIN_TAIL = [
    "Nigeria", "Bangladesh", "Nepal", "Egypt", "Vietnam", "Czech Republic", "Ghana",
    "Cameroon", "Morocco", "Indonesia", "Brazil", "Sri Lanka", "Jordan", "Tunisia",
    "Ukraine", "Italy", "Spain", "Colombia", "Mexico", "Kenya", "Ethiopia", "Philippines",
    "Ecuador", "Poland", "Greece",
]

# World regions used as the first generalisation level for nationality (k-anonymity).
REGION = {
    "Germany": "Germany",
    "Austria": "EU/EEA", "Czech Republic": "EU/EEA", "Italy": "EU/EEA", "Spain": "EU/EEA",
    "Poland": "EU/EEA", "Greece": "EU/EEA",
    "Türkiye": "Europe (non-EU)", "Ukraine": "Europe (non-EU)",
    "India": "South Asia", "Pakistan": "South Asia", "Bangladesh": "South Asia",
    "Nepal": "South Asia", "Sri Lanka": "South Asia",
    "China": "East/SE Asia", "Vietnam": "East/SE Asia", "Indonesia": "East/SE Asia",
    "Philippines": "East/SE Asia",
    "Iran": "Middle East/N. Africa", "Syria": "Middle East/N. Africa", "Egypt": "Middle East/N. Africa",
    "Morocco": "Middle East/N. Africa", "Jordan": "Middle East/N. Africa", "Tunisia": "Middle East/N. Africa",
    "Nigeria": "Sub-Saharan Africa", "Ghana": "Sub-Saharan Africa", "Cameroon": "Sub-Saharan Africa",
    "Kenya": "Sub-Saharan Africa", "Ethiopia": "Sub-Saharan Africa",
    "Brazil": "Latin America", "Colombia": "Latin America", "Mexico": "Latin America",
    "Ecuador": "Latin America",
}

# Female share: Bavarian engineering entrants 28.6 % (2025), graduates ~26 % [S4].
# ASSUMPTION: 25 % female; 0.3 % "diverse" (legal category since 2018, not in the cited stats).
GENDER = {"male": 0.747, "female": 0.25, "diverse": 0.003}

# Age. Mean student age 25.8 (all), 24.9 full-time [S5]; median 24 at first degree [S6].
# Master's age not published: ASSUMPTION mean ~26.5 (22 + Gamma(2, 2.25)).
BACHELOR_ENTRY_AGE = 19
MASTER_AGE_BASE, MASTER_AGE_SHAPE, MASTER_AGE_SCALE = 22, 2.0, 2.25

# German grading scale 1.0 (best) .. 4.0 (lowest pass) [S7]. Engineering mean 2.1-2.2,
# Master better than Bachelor (~1.8 vs ~2.2) [S7, Prüfungsjahr 2010, old].
GRADE_MEAN = {"Bachelor": 2.2, "Master": 1.9}
GRADE_SD = {"Bachelor": 0.5, "Master": 0.45}

# Special-category data (GDPR Art. 9): students with a study-impairing health condition.
# 22nd Sozialerhebung reports ~16 % [S5] (figure to verify against the report).
DISABILITY_SUPPORT_SHARE = 0.16

# Postcodes (real town/postcode pairs). Weights are an ASSUMPTION: domestic students come mainly
# from Landkreis Cham and eastern Bavaria; international students live near campus.
POSTCODES_DOMESTIC = {
    "93413": ("Cham", 0.22), "93437": ("Furth im Wald", 0.08), "93444": ("Bad Kötzting", 0.08),
    "93449": ("Waldmünchen", 0.05), "93426": ("Roding", 0.08), "93047": ("Regensburg", 0.12),
    "94315": ("Straubing", 0.07), "94469": ("Deggendorf", 0.07), "92421": ("Schwandorf", 0.07),
    "92637": ("Weiden in der Oberpfalz", 0.05), "92224": ("Amberg", 0.04), "94032": ("Passau", 0.03),
    "90402": ("Nürnberg", 0.02), "80331": ("München", 0.02),
}
POSTCODES_INTERNATIONAL = {
    "93413": ("Cham", 0.80), "93437": ("Furth im Wald", 0.06), "93426": ("Roding", 0.06),
    "93047": ("Regensburg", 0.08),
}
