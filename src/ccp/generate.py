"""Fake Campus Cham data: a student table and free-text helpdesk/feedback tickets.

No real person is in the output. Distributions follow ccp.calibration (public statistics plus
marked assumptions). Every personal-data string inserted into a ticket is recorded as a
ground-truth span, so text anonymisation can be scored for recall and leaks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from . import calibration as C
from .names import NamePools

STUDENT_COLUMNS = [
    "matriculation_no", "first_name", "last_name", "email", "phone", "birth_date", "street",
    "postcode", "city", "age", "gender", "nationality", "programme", "degree", "enrolment_year",
    "semester", "ects", "avg_grade", "disability_support", "at_risk",
]
DIRECT_IDENTIFIERS = ["matriculation_no", "first_name", "last_name", "email", "phone", "birth_date", "street", "city"]
QUASI_IDENTIFIERS = ["age", "gender", "nationality", "programme", "enrolment_year", "postcode"]
SENSITIVE = ["avg_grade", "disability_support", "at_risk", "ects", "semester"]

GRADE_STEPS = np.array([1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0])


def _ascii(s: str) -> str:
    table = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})
    import unicodedata

    s = unicodedata.normalize("NFKD", s.translate(table)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9.-]", "", s.lower().replace(" ", "-"))


def _phone(rng: np.random.Generator) -> str:
    prefix = rng.choice(["151", "152", "157", "160", "162", "170", "172", "176", "178"])
    body = "".join(rng.choice(list("0123456789"), 7 + int(rng.integers(0, 2))))
    style = rng.integers(0, 4)
    if style == 0:
        return f"+49 {prefix} {body}"
    if style == 1:
        return f"0{prefix} {body[:4]} {body[4:]}"
    if style == 2:
        return f"0{prefix}/{body}"
    return f"+49{prefix}{body}"


def generate_students(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng([seed, 0])  # distinct stream per purpose: [seed, 0] students, [seed, 1] tickets
    codes = list(C.PROGRAMMES)
    shares = np.array([C.PROGRAMMES[c][3] for c in codes])
    programme = rng.choice(codes, n, p=shares / shares.sum())
    degree = np.array([C.PROGRAMMES[p][1] for p in programme])
    std_sem = np.array([C.PROGRAMMES[p][4] for p in programme])

    # Semester: uniform over the standard period plus a tail of students over the standard time.
    overtime = rng.random(n) < 0.15
    semester = np.where(overtime, std_sem + rng.integers(1, 4, n), rng.integers(1, std_sem + 1))
    # Winter term: odd semesters started in a winter term of year CURRENT_YEAR - (s-1)//2.
    enrolment_year = C.CURRENT_YEAR - (semester - 1) // 2

    intl_p = np.where(degree == "Bachelor", C.INTERNATIONAL_SHARE["Bachelor"], C.INTERNATIONAL_SHARE["Master"])
    international = rng.random(n) < intl_p
    head = np.array(list(C.ORIGIN_SOURCED.values()))
    tail_w = 1.0 / np.sqrt(np.arange(1, len(C.ORIGIN_TAIL) + 1))  # slowly decaying long tail (assumption)
    tail_w = tail_w / tail_w.sum() * (1 - head.sum())
    origins = np.array(list(C.ORIGIN_SOURCED) + C.ORIGIN_TAIL)
    origin_p = np.concatenate([head, tail_w])
    nationality = np.where(international, rng.choice(origins, n, p=origin_p / origin_p.sum()), "Germany")

    gender = rng.choice(list(C.GENDER), n, p=list(C.GENDER.values()))

    master_age = C.MASTER_AGE_BASE + rng.gamma(C.MASTER_AGE_SHAPE, C.MASTER_AGE_SCALE, n) + (semester - 1) / 2
    bachelor_age = C.BACHELOR_ENTRY_AGE + rng.gamma(1.5, 1.0, n) + (semester - 1) / 2
    age = np.clip(np.where(degree == "Master", master_age, bachelor_age), 18, 49).astype(int)
    birth_year = C.CURRENT_YEAR - age - (rng.random(n) < 0.8)  # most birthdays already passed by November
    birth_date = pd.to_datetime(
        {"year": birth_year, "month": rng.integers(1, 13, n), "day": rng.integers(1, 29, n)}
    ).dt.strftime("%Y-%m-%d").to_numpy()

    def pick_postcodes(table: dict[str, tuple[str, float]], size: int) -> np.ndarray:
        keys = list(table)
        w = np.array([table[k][1] for k in keys])
        return rng.choice(keys, size, p=w / w.sum())

    postcode = np.where(
        international,
        pick_postcodes(C.POSTCODES_INTERNATIONAL, n),
        pick_postcodes(C.POSTCODES_DOMESTIC, n),
    )
    town = {**{k: v[0] for k, v in C.POSTCODES_DOMESTIC.items()}, **{k: v[0] for k, v in C.POSTCODES_INTERNATIONAL.items()}}
    city = np.array([town[p] for p in postcode])

    mean = np.where(degree == "Master", C.GRADE_MEAN["Master"], C.GRADE_MEAN["Bachelor"])
    sd = np.where(degree == "Master", C.GRADE_SD["Master"], C.GRADE_SD["Bachelor"])
    latent = rng.normal(0, 1, n)  # shared ability factor ties grade, credits and risk together
    raw_grade = np.clip(mean + sd * (0.8 * latent + 0.6 * rng.normal(0, 1, n)), 1.0, 4.0)
    avg_grade = np.round(raw_grade, 1)

    disability = rng.random(n) < C.DISABILITY_SUPPORT_SHARE
    # Academic-risk flag: worse grades, overtime and (weakly) disability raise the odds.
    logit = -2.0 + 1.8 * (raw_grade - mean) / sd + 1.0 * overtime + 0.4 * disability
    at_risk = rng.random(n) < 1 / (1 + np.exp(-logit))
    expected = np.minimum(semester - 1, std_sem) * 30
    # Higher latent = worse grade, so it lowers credit completion.
    completion = np.clip(0.9 - 0.08 * latent - 0.35 * at_risk + rng.normal(0, 0.08, n), 0.2, 1.0)
    ects = (np.round(expected * completion / 5) * 5).astype(int)

    names = NamePools(sorted(set(nationality)), seed)
    first, last = names.sample(rng, nationality, gender)
    matno = (rng.choice(20_000_000, n, replace=False) + 10_000_000).astype(str)
    email_local = np.array([f"{_ascii(f)}.{_ascii(l)}" for f, l in zip(first, last)], dtype=object)
    # Duplicate local parts get a number, like real university mail systems.
    dup = pd.Series(email_local).groupby(email_local).cumcount().to_numpy()
    email = np.array([f"{e}{d if d else ''}@stud.th-deg.de" for e, d in zip(email_local, dup)])
    phone = np.array([_phone(rng) for _ in range(n)])
    fk = Faker("de_DE")
    fk.seed_instance(seed)
    streets = np.array([fk.street_name() for _ in range(300)])
    street = np.char.add(np.char.add(rng.choice(streets, n).astype(str), " "), rng.integers(1, 80, n).astype(str))

    df = pd.DataFrame({
        "matriculation_no": matno, "first_name": first, "last_name": last, "email": email,
        "phone": phone, "birth_date": birth_date, "street": street, "postcode": postcode, "city": city,
        "age": age, "gender": gender, "nationality": nationality, "programme": programme,
        "degree": degree, "enrolment_year": enrolment_year, "semester": semester, "ects": ects,
        "avg_grade": avg_grade, "disability_support": disability, "at_risk": at_risk,
    })
    return df[STUDENT_COLUMNS]


# --- free text -------------------------------------------------------------------------------
# Slots: {full} {first} {lower} {email} {matno} {phone} {address} {dob} {iban} {staff} {qi}
# {prog} and {nat} are not personal on their own; {qi} is an indirect identifier phrase.
TEMPLATES_EN = [
    "Hello, my name is {full} (matriculation number {matno}). I cannot access iLearn for {prog} since Monday. Please call me on {phone}.",
    "Hi IT team, {first} here. My password reset mail never arrived at {email}. Thanks!",
    "Dear Student Office, please update my address to {address}. Matriculation no. {matno}. Best regards, {full}",
    "Feedback on the lab course: the robot cell was broken twice. I am {qi}, so the group work was hard for me. - {first}",
    "I need a confirmation of enrolment for my visa extension. Name: {full}, date of birth {dob}, matriculation {matno}.",
    "Please refund the semester fee to my account {iban}. Account holder: {full}.",
    "{staff} told me to contact you about my exam registration. My email is {email} and my phone is {phone}.",
    "hi, this is {lower}, my student id is {matno}. the printer in the sensor lab does not work",
    "My room in the dormitory at {address} has no internet. Contact: {phone} ({first})",
    "As {qi}, I feel that the course materials assume prior knowledge from German universities.",
    "Could you send the thesis guideline to {email}? I am writing my thesis with {staff}. Regards, {full}",
    "Request to extend the deadline for the project report. Student: {full}, {matno}, born {dob}.",
]
TEMPLATES_DE = [
    "Hallo, ich bin {full} (Matrikelnummer {matno}) und komme seit Montag nicht mehr in iLearn. Bitte rufen Sie mich unter {phone} an.",
    "Sehr geehrte Damen und Herren, bitte ändern Sie meine Adresse auf {address}. Mit freundlichen Grüßen, {full}",
    "Hi, hier ist {first}. Die Mail zum Passwort-Reset kam nie bei {email} an.",
    "Feedback zur Vorlesung bei {staff}: zu viel Stoff in zu wenig Zeit. Ich bin {qi}.",
    "Ich brauche eine Immatrikulationsbescheinigung. Name: {full}, geboren am {dob}, Matr.-Nr. {matno}.",
    "Bitte überweisen Sie die Rückerstattung auf mein Konto {iban}, Kontoinhaber {full}.",
    "servus, {lower} hier, meine matrikelnummer ist {matno}, der drucker im labor geht nicht",
    "Meine Handynummer hat sich geändert, neu: {phone}. Gruß {first} {matno}",
    "Als {qi} finde ich die Prüfungsorganisation unübersichtlich.",
    "Könnten Sie mir den Leitfaden an {email} schicken? Ich schreibe meine Abschlussarbeit bei {staff}.",
]

LABEL = {
    "full": "PERSON", "first": "PERSON", "lower": "PERSON", "staff": "PERSON", "email": "EMAIL_ADDRESS",
    "matno": "MATRICULATION_NO", "phone": "PHONE_NUMBER", "address": "LOCATION", "dob": "DATE_OF_BIRTH",
    "iban": "IBAN_CODE", "qi": "INDIRECT_IDENTIFIER",
}
_SLOT = re.compile(r"\{(\w+)\}")


@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str


def _fill(template: str, values: dict[str, str]) -> tuple[str, list[Span]]:
    out, spans, pos = [], [], 0
    length = 0
    for m in _SLOT.finditer(template):
        lit = template[pos:m.start()]
        out.append(lit)
        length += len(lit)
        val = values[m.group(1)]
        if m.group(1) in LABEL:
            spans.append(Span(length, length + len(val), LABEL[m.group(1)], val))
        out.append(val)
        length += len(val)
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def _ticket_streams(seed: int) -> tuple[list[str], np.random.Generator, Faker]:
    """Random streams for the tickets, with the 25 staff names drawn first. staff_roster and
    generate_tickets share this, so the staff register is exactly the set of names in the tickets."""
    rng = np.random.default_rng([seed, 1])
    fk = Faker("de_DE")
    fk.seed_instance(seed + 1)
    staff = [f"{rng.choice(['Prof. Dr.', 'Dr.', 'Frau', 'Herr'])} {fk.last_name()}" for _ in range(25)]
    return staff, rng, fk


def staff_roster(seed: int) -> list[str]:
    """Staff names (title + surname) that generate_tickets(..., seed) mentions: the staff register."""
    return _ticket_streams(seed)[0]


def generate_tickets(students: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    staff, rng, fk = _ticket_streams(seed)
    nat_counts = students.groupby(["nationality", "programme"]).size()
    rows = []
    for i in range(n):
        s = students.iloc[int(rng.integers(len(students)))]
        # Language: German-taught Bachelor and German nationals mostly write German.
        p_de = 0.85 if (s.degree == "Bachelor" or s.nationality == "Germany") else 0.1
        lang = "de" if rng.random() < p_de else "en"
        tpl = rng.choice(TEMPLATES_DE if lang == "de" else TEMPLATES_EN)
        dob = pd.Timestamp(s.birth_date)
        rare = nat_counts[(s.nationality, s.programme)] <= 2
        if lang == "en":
            qi = (f"the only student from {s.nationality} in {s.programme}" if rare
                  else f"a {s.age}-year-old {s.gender} student from {s.nationality} in {s.programme}")
            dob_s = dob.strftime("%d.%m.%Y") if rng.random() < 0.5 else dob.strftime("%B %d, %Y")
        else:
            qi = (f"die einzige Person aus {s.nationality} im Studiengang {s.programme}" if rare
                  else f"{s.age} Jahre alt, aus {s.nationality}, im Studiengang {s.programme}")
            dob_s = dob.strftime("%d.%m.%Y")
        values = {
            "full": f"{s.first_name} {s.last_name}", "first": s.first_name,
            "lower": f"{s.first_name} {s.last_name}".lower(), "email": s.email, "matno": s.matriculation_no,
            "phone": s.phone, "address": f"{s.street}, {s.postcode} {s.city}", "dob": dob_s,
            "iban": fk.iban(), "staff": str(rng.choice(staff)), "qi": qi,
            "prog": C.PROGRAMMES[s.programme][0], "nat": s.nationality,
        }
        text, spans = _fill(str(tpl), values)
        rows.append({
            "ticket_id": f"T{i:06d}", "matriculation_no": s.matriculation_no, "lang": lang, "text": text,
            "spans": json.dumps([sp.__dict__ for sp in spans], ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def write(out_dir: Path, n_students: int, n_tickets: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    students = generate_students(n_students, seed)
    tickets = generate_tickets(students, n_tickets, seed)
    students.to_csv(out_dir / "students.csv", index=False)
    tickets.to_csv(out_dir / "tickets.csv", index=False)
    return students, tickets
