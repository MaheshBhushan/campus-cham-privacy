# Calibration parameters: synthetic THD Campus Cham student dataset
All sources accessed 2026-11-04. Confidence: **H** = official figure read directly; **M** = official figure but proxy population (Germany/Bavaria rather than Cham) or older year; **A** = assumption/derivation, no direct source.

## 1. Programmes at Campus Cham (WS 2025/26 / 2026)
Source: THD, "Campus Cham – Study programmes" (https://www.th-deg.de/en/campus-cham) [S1]

| Code (dataset) | Programme | Degree | Language |
|---|---|---|---|
| MDP (Bachelor) | Mechatronik & Projektmanagement für Digitale Produktion | B.Eng. | German |
| MMC | Mechatronic and Cyber-Physical Systems | M.Eng. | English |
| MSS | Artificial Intelligence for Smart Sensors and Actuators | M.Eng. | English |
| **MDM** | **Applied AI for Digital Production Management** (URL slug `dm-m-en`) | M.Eng. | English |
| **MRO** | **Robotics** (URL slug `mro-m-en`) | M.Eng. | English |

THD's own WS 2025/26 fact sheet lists "Lehrstandort Campus Cham" with **1 Bachelor + 4 Master programmes** [S2] – consistent. Note: the code-to-name mapping for MDM is inferred from URL slugs (`dm`) – confidence M; MRO is explicit in the slug – H.

## 2. Calibration table

| # | Parameter | Value | Source | Year | Conf. |
|---|---|---|---|---|---|
| 1 | THD total students | 9,624 (ordinary + on leave, incl. guest/IAS, excl. doctoral) | THD Zahlen-Daten-Fakten WS 25/26 [S2] | WS 2025/26 (cut-off 01.12.2025) | H |
| 2 | Campus Cham students (total) | **913** | [S2] | WS 2025/26 | H |
| 3 | Campus Cham students per programme | **Not published** – split must be assumed (e.g. proportional). Suggested assumption: Bachelor MDP minority, 4 Masters majority, given 4 of 5 programmes are English-taught Masters | — | — | A |
| 4 | Campus Cham staff | 9 professors, 35 employees | [S2] | WS 2025/26 | H |
| 5 | THD international students | 4,940 = **51.3 %** of students (excl. doctoral) | [S2] | WS 2025/26 | H |
| 6 | THD first-year students | 2,833 (1st subject semester); 2,073 (1st university semester) | [S2] | WS 2025/26 | H |
| 7 | Campus Cham international share | **Not published as %.** THD states "roughly 1,000 fellow students from more than 50 nations"; regional page says "over 40 nations". Assume ≥ THD average (51 %) because 4/5 programmes are English Masters; ~70–85 % is a plausible assumption, **not sourced** | [S1], [S8] | 2026 | A |
| 8 | Top countries of origin, internat. students in Germany | India 58,833 (14.6 %), China 38,989 (9.7 %), Türkiye 20,900 (5.2 %), Iran 17,248 (4.3 %), Austria 15,732 (3.9 %), Pakistan 12,646 (3.1 %), Syria 11,639 (2.9 %) | DAAD/DZHW Wissenschaft weltoffen kompakt 2026 [S3] | WS 2024/25 | H (national) / M as proxy for Cham |
| 9 | Internat. students Germany total; at HAW | 402,083 total; 131,726 at HAW | [S3] | WS 2024/25 | H |
| 10 | Degree aimed for by internat. students (Germany) | ~35 % Bachelor, 47 % Master | [S3] | WS 2024/25 | H |
| 11 | THD-specific country breakdown | **Not found** in public THD/LfStat sources → use [S3] national distribution as proxy | — | — | A |
| 12 | Female share, engineering, 1st-semester entrants, all Bavarian HEIs | 28.8 % (2024), 28.6 % (2025) | Bayerischer Landtag Drs. 19/13247 (data: Bayer. Landesamt für Statistik/CEUS) [S4] | Studienjahr 2024/2025 | H (Bavaria) / M as proxy for HAW Cham |
| 13 | Female share, engineering graduates, Bavaria | Bachelor 26.1 %, Master 25.7 % | [S4] | Prüfungsjahr 2024 | H |
| 14 | → Suggested gender parameter for Cham | ~25 % female, ~75 % male (no non-binary category in official stats) | derived from 12–13 | — | A |
| 15 | Mean age, all students in Germany (survey) | 25.8 years overall; full-time on-campus ("Präsenz") 24.9 | 22. Sozialerhebung (DZHW) [S5] | 2021 | H |
| 16 | Median age at first degree (≈90 % Bachelor), NRW | 24 years (MINT 23) | IT.NRW press release 267/24 [S6] | Prüfungsjahr 2023 | M (NRW proxy) |
| 17 | Master's student age | **No direct Destatis/DZHW figure retrieved.** Assumption: Bachelor completion ~24 + Master 3–4 semesters → domestic Master students ~24–28; international Masters typically older (prior work experience) – suggest mean ≈ 26–27, range 22–35 | derived from 15–16 | — | A |
| 18 | German grading scale | 1.0–5.0; passing steps 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0; 4.0 = lowest pass; 5.0 = fail | Standard German HEI scale, as used in Wissenschaftsrat [S7]; confirm against THD APO | current | H (well-established) |
| 19 | Mean final grade, engineering (all degree types) | 2.1–2.2 | Wissenschaftsrat, Prüfungsnoten an Hochschulen im Prüfungsjahr 2010 [S7] | 2000–2011 | M (old) |
| 20 | Grade tendency by degree | Master grades better than Bachelor (Master FH, all subjects ≈ 1.8; Bachelor FH engineering ≈ 2.2) | [S7] Tab. (Prüfungsjahr 2010) – table text partially garbled in extraction | 2010 | M |
| 21 | Germany graduates context | 527,600 graduates; Bachelor 49 %, Master 29 %; engineering 25 % of degrees; 53 % female overall | Destatis PM 352/2026 [S9] | Prüfungsjahr 2025 | H |

## 3. Campus Cham context facts
- Addresses: Badstraße 21 and Dr.-Muggenthaler-Straße 11, 93413 Cham [S1].
- Youngest THD study location; ~1,000 students, >50 nations [S1]; 913 students officially WS 25/26 [S2].
- Research: Technology transfer centres "Intelligent Production" (founded **2010**) and "Intelligent Robotics"; Digital Innovation Centre, IHK and Mechatronics Network Eastern Bavaria on site [S1], [S10].
- Labs: Electrical Engineering, Embedded/IoT, IT, Sensor Lab, Intelligent Robotics Lab [S1].
- Town ~17,000 inhabitants; "close to the Czech Republic"; Prague/Munich/Nuremberg airports ≈2 h [S1]. (Exact km to border not sourced.)
- "Living Lab": **not found** on official Cham pages → do not use without a source.

## 4. Sources (reference list)
- [S1] Technische Hochschule Deggendorf (2026). *Campus Cham – Teaching Campus Cham / Studying at Campus Cham*. https://www.th-deg.de/en/campus-cham (accessed 2026-11-04).
- [S2] Technische Hochschule Deggendorf (2025). *Zahlen, Daten, Fakten – Profil der Hochschule im Wintersemester 2025/2026* (PRIMUSS-Stichtag 01.12.2025). https://www.th-deg.de/Hochschule/profil%20der%20hochschule/zahlen-daten-fakten-ws-25-26_2.pdf (accessed 2026-11-04).
- [S3] DAAD & DZHW (2026). *Wissenschaft weltoffen kompakt 2026. Daten und Fakten zur Internationalität von Studium und Forschung in Deutschland und weltweit*. Bielefeld: wbv. https://www.wissenschaft-weltoffen.de/content/uploads/2026/05/wiwe_kompakt_2026_bf_DE.pdf (accessed 2026-11-04).
- [S4] Bayerischer Landtag (2026). *Schriftliche Anfrage der Abgeordneten Dr. Simone Strohmayr (SPD) vom 30.06.2026 „Frauen in Ingenieurberufen" und Antwort der Staatsregierung*, Drucksache 19/13247 (data: Bayerisches Landesamt für Statistik/CEUS). https://www.bayern.landtag.de/www/ElanTextAblage_WP19/Drucksachen/Schriftliche%20Anfragen/19_0013247.pdf (accessed 2026-11-04).
- [S5] Kroher, M., Beuße, M., Isleib, S., Becker, K., Ehrhardt, M.-C., Gerdes, F., Koopmann, J., Schommer, T., Schwabe, U., Steinkühler, J., Völk, D., Peter, F. & Buchholz, S. (2023). *Die Studierendenbefragung in Deutschland: 22. Sozialerhebung. Die wirtschaftliche und soziale Lage der Studierenden in Deutschland 2021*. Berlin: BMBF / Hannover: DZHW. https://www.dzhw.eu/pdf/ab_20/Soz22_Hauptbericht.pdf (accessed 2026-11-04).
- [S6] IT.NRW (2024, 23 Aug). *NRW-Hochschulen: Studierende sind bei Abschluss ihres Erststudiums durchschnittlich 24 Jahre alt* (Pressemitteilung 267/24). https://www.it.nrw/nrw-hochschulen-studierende-sind-bei-abschluss-ihres-erststudiums-durchschnittlich-24-jahre-alt (accessed 2026-11-04).
- [S7] Wissenschaftsrat (2012). *Prüfungsnoten an Hochschulen im Prüfungsjahr 2010. Arbeitsbericht mit einem wissenschaftspolitischen Kommentar* (Drs. 2627-12). Hamburg. https://www.wissenschaftsrat.de/download/archiv/2627-12.pdf (accessed 2026-11-04).
- [S8] Aktionskreis Lebens- und Wirtschaftsraum Landkreis Cham e.V. (n.d.). *Technologie Campus Cham*. https://www.aktionskreis-cham.de/regionale-projekte/technologie-campus-cham (accessed 2026-11-04).
- [S9] Statistisches Bundesamt (Destatis) (2026, 2 Oct). *Zahl der Hochschulabschlüsse 2025 um 3 % gestiegen* (Pressemitteilung Nr. 352). https://www.destatis.de/DE/Presse/Pressemitteilungen/2026/10/PD26_352_213.html (accessed 2026-11-04). Detailed tables: GENESIS 21321-0001 to -0007.
- [S10] Technische Hochschule Deggendorf (n.d.). *Campus Cham – Intelligente Produktion*. https://www.th-deg.de/campus-cham-intelligente-produktion (accessed 2026-11-04).

## 5. Gaps / not verified
- Per-programme headcount at Cham; Cham international share and nationality mix; THD nationality breakdown (try THD controlling request or LfStat Bayern GENESIS 21311 per-Hochschule tables – not retrieved).
- Master's-student age (Destatis "Prüfungen an Hochschulen" has graduate age by degree – not retrieved).
- Recent Wissenschaftsrat grade report (newer than Prüfungsjahr 2010) – not located; values above are old.
- Destatis national female share for engineering at HAW specifically – not retrieved; Bavarian all-HEI figures used.
