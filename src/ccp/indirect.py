"""Indirect identifiers in free text: coarsening rules, an anonymity-set metric and an LLM rewrite.

Presidio removes names, e-mails and numbers, but a self-description such as "a 26-year-old female
student from Iran in MMC" or "die einzige Person aus Vietnam im Studiengang MDP" survives in parts:
the country may become <LOCATION> while age, gender and programme stay. Together these
quasi-identifiers (QIs) can still single the writer out.

This module runs after Presidio on `text_anonymised`:

- `generalise_indirect` applies the k-anonymity idea to the sentence itself, with EN/DE lexicons:
  country -> world region (calibration.REGION), exact age -> decade ("in their twenties" /
  "in den Zwanzigern"), programme code or name -> degree ("a Master's programme" /
  "einem Masterstudiengang"), gender words are dropped and "the only" / "die einzige" become "a" /
  "eine". Texts without a self-description are returned unchanged, so a programme named as the
  topic of a ticket ("iLearn for Robotics") keeps its meaning.
- `anonymity_set_sizes` is the metric: it reads back what an attacker still learns about the
  writer's QIs from the text (country or region, exact age or decade, programme or degree, gender)
  and counts the students in the table who match all of it. A set of size 1 means the writer is
  singled out even though no name is left.
- `llm_rewrite` is the alternative: a small instruction-tuned model (Qwen2.5, run locally with
  transformers on CPU, so the tickets never leave the machine) rewrites only the texts that contain
  a self-description. It is slower, less predictable and can drop parts of the message; the
  rule-based version is the default.
"""

from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

from . import calibration as C

# --- lexicons -----------------------------------------------------------------------------------
# Surface forms (EN as generate.py writes them, common EN variants, German names) -> REGION key.
_COUNTRY_DE = {
    "Germany": ["Deutschland"], "Austria": ["Österreich"],
    "Czech Republic": ["Czechia", "Tschechien", "Tschechische Republik", "Tschechischen Republik"],
    "Italy": ["Italien"], "Spain": ["Spanien"], "Poland": ["Polen"], "Greece": ["Griechenland"],
    "Türkiye": ["Turkey", "Türkei"], "Ukraine": [], "India": ["Indien"], "Pakistan": [],
    "Bangladesh": ["Bangladesch"], "Nepal": [], "Sri Lanka": [], "China": [], "Vietnam": [],
    "Indonesia": ["Indonesien"], "Philippines": ["Philippinen"], "Iran": [], "Syria": ["Syrien"],
    "Egypt": ["Ägypten"], "Morocco": ["Marokko"], "Jordan": ["Jordanien"], "Tunisia": ["Tunesien"],
    "Nigeria": [], "Ghana": [], "Cameroon": ["Kamerun"], "Kenya": ["Kenia"], "Ethiopia": ["Äthiopien"],
    "Brazil": ["Brasilien"], "Colombia": ["Kolumbien"], "Mexico": ["Mexiko"], "Ecuador": [],
}
COUNTRY_ALIASES = {alias: c for c in C.REGION for alias in [c, *_COUNTRY_DE[c]]}

# Region -> (EN after "from", EN bare, DE after "aus", DE bare). Germany stays as it is: domestic
# students are the large majority of the Bachelor, so "Germany" is already a coarse value.
REGION_PHRASES = {
    "Germany": ("Germany", "Germany", "Deutschland", "Deutschland"),
    "EU/EEA": ("another EU country", "another EU country", "einem anderen EU-Land", "ein anderes EU-Land"),
    "Europe (non-EU)": ("a European country outside the EU", "a European country outside the EU",
                        "einem europäischen Land außerhalb der EU", "ein europäisches Land außerhalb der EU"),
    "South Asia": ("South Asia", "South Asia", "Südasien", "Südasien"),
    "East/SE Asia": ("East or Southeast Asia", "East or Southeast Asia", "Ost- oder Südostasien",
                     "Ost- oder Südostasien"),
    "Middle East/N. Africa": ("the Middle East or North Africa", "the Middle East or North Africa",
                              "dem Nahen Osten oder Nordafrika", "der Nahe Osten oder Nordafrika"),
    "Sub-Saharan Africa": ("Sub-Saharan Africa", "Sub-Saharan Africa", "Subsahara-Afrika", "Subsahara-Afrika"),
    "Latin America": ("Latin America", "Latin America", "Lateinamerika", "Lateinamerika"),
}
_ABROAD = r"abroad|another country|international student|(?:dem )?Ausland|einem anderen Land"

DECADE_EN = {1: "in their teens", 2: "in their twenties", 3: "in their thirties", 4: "in their forties"}
DECADE_DE = {1: "im Teenageralter", 2: "in den Zwanzigern", 3: "in den Dreißigern", 4: "in den Vierzigern"}
_DECADE_WORDS = {"teens": 1, "twenties": 2, "thirties": 3, "forties": 4, "teenageralter": 1,
                 "zwanzigern": 2, "dreißigern": 3, "vierzigern": 4}

DEGREE_EN = {"Bachelor": "a Bachelor's programme", "Master": "a Master's programme"}
DEGREE_DE = {"Bachelor": "Bachelorstudiengang", "Master": "Masterstudiengang"}
PROGRAMME_ALIASES = {**{c: c for c in C.PROGRAMMES}, **{v[0]: c for c, v in C.PROGRAMMES.items()}}

GENDER_WORDS = {"female": "female", "woman": "female", "weiblich": "female", "studentin": "female",
                "male": "male", "männlich": "male", "diverse": "diverse", "divers": "diverse",
                "non-binary": "diverse"}


def _alt(words: list[str]) -> str:
    # Longest first so "Czech Republic" wins over a shorter overlapping alias.
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_COUNTRY = _alt(list(COUNTRY_ALIASES))
_PROG = _alt(list(PROGRAMME_ALIASES))
_REGION_TEXT = {p: r for r, ps in REGION_PHRASES.items() for p in ps if p not in COUNTRY_ALIASES}
_NOUN_EN = r"student|person|woman"
_NOUN_DE = r"Person|Student\w*|Studierende\w*"

RE_COUNTRY = re.compile(rf"\b(?:{_COUNTRY})\b")
RE_REGION = re.compile(rf"(?<!\w)(?:{_alt(list(_REGION_TEXT))})(?!\w)")
RE_ABROAD = re.compile(rf"\b(?:{_ABROAD})\b", re.IGNORECASE)
RE_AGE = re.compile(r"\b(\d{2})(?:[- ]years?[- ]old\b|\s+Jahre\s+alt\b|-jährige?[rnms]?\b)|\baged\s+(\d{2})\b")
RE_DECADE = re.compile(r"\bin (?:their|my|his|her) (?:late |early |mid-)?(teens|twenties|thirties|forties)\b"
                       r"|\b(?:im|in den) (Teenageralter|Zwanzigern|Dreißigern|Vierzigern)\b")
RE_PROG = re.compile(rf"(?<!\w)(?:{_PROG})(?!\w)")
RE_DEGREE = re.compile(r"\b(Bachelor|Master)")
# Gender adjectives only next to a person noun: "diverse"/"divers" also mean "varied".
RE_GENDER = re.compile(rf"\b(female|male|diverse|non-binary) (?:{_NOUN_EN})\b|\b(woman|Studentin)\b"
                       rf"|\b(weiblich|männlich|divers)e?[rnms]? (?:{_NOUN_DE})\b", re.IGNORECASE)
RE_ONLY = re.compile(rf"\b(?:the only (?:\w+ )?(?:{_NOUN_EN}|one)|(?:(?:die|der) )?einzigen? (?:\w+ )?(?:{_NOUN_DE}))\b",
                     re.IGNORECASE)


def has_indirect(text: str) -> bool:
    """A self-description: age, origin, gender or a uniqueness claim ("the only ..."). A programme
    on its own is usually the topic of the ticket, not a description of the writer."""
    return any(r.search(text) for r in (RE_ONLY, RE_AGE, RE_DECADE, RE_COUNTRY, RE_REGION, RE_ABROAD, RE_GENDER))


# --- rule-based coarsening ----------------------------------------------------------------------
def _decade(age: str, lang: str) -> str:
    d = min(max(int(age) // 10, 1), 4)
    return (DECADE_DE if lang == "de" else DECADE_EN)[d]


def _region(country: str) -> str:
    return C.REGION[COUNTRY_ALIASES[country]]


def _degree(prog: str) -> str:
    return C.PROGRAMMES[PROGRAMME_ALIASES[prog]][1]


def _keep_case(repl: str, original: str) -> str:
    return repl[:1].upper() + repl[1:] if original[:1].isupper() else repl


def _generalise_en(t: str) -> str:
    t = re.sub(r"\bthe only one\b", lambda m: _keep_case("one", m.group()), t, flags=re.IGNORECASE)
    t = re.sub(rf"\bthe only (?=(?:\w+ )?(?:{_NOUN_EN})\b)", lambda m: _keep_case("a ", m.group()), t,
               flags=re.IGNORECASE)
    t = re.sub(rf"\b(?:female|male|diverse|non-binary) (?=(?:{_NOUN_EN})\b)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bwoman\b", "person", t)
    t = re.sub(r"\bI(?:'m| am) (\d{2})(?: years? old)?\b", lambda m: "I am " + _decade(m[1], "en").replace("their", "my"), t)
    t = re.sub(rf"\b(\d{{2}})[- ]year[- ]old ({_NOUN_EN})\b", lambda m: f"{m[2]} {_decade(m[1], 'en')}", t)
    t = re.sub(r"\b(\d{2})[- ]years?[- ]old\b|\baged (\d{2})\b", lambda m: _decade(m[1] or m[2], "en"), t)
    t = re.sub(rf"\bfrom (?:the )?({_COUNTRY})\b", lambda m: "from " + REGION_PHRASES[_region(m[1])][0], t)
    t = RE_COUNTRY.sub(lambda m: REGION_PHRASES[_region(m.group())][1], t)
    t = re.sub(rf"\b(?:the )?({_PROG})(?: programme| program)?(?!\w)",
               lambda m: DEGREE_EN[_degree(m[1])], t)
    return t


def _generalise_de(t: str) -> str:
    person = rf"(?=(?:\w+ )?(?:{_NOUN_DE})\b)"
    t = re.sub(rf"\b([Dd])ie einzige {person}", lambda m: _keep_case("eine ", m[1]), t)
    t = re.sub(rf"\b([Dd])er einzige {person}", lambda m: _keep_case("ein ", m[1]), t)
    t = re.sub(rf"\beinzigen? {person}", "", t)
    t = re.sub(rf"\b(?:weiblich|männlich|divers)e[rnms]? (?=(?:{_NOUN_DE})\b)", "", t)
    t = re.sub(r"\bStudentin\b", "Person", t)
    t = re.sub(r"\b(\d{2})-jährige?[rnms]? (\w+)", lambda m: f"{m[2]} {_decade(m[1], 'de')}", t)
    t = re.sub(r"\b([Aa]ls) (?=\d{2} Jahre alt\b)", r"\1 Person ", t)  # "Als 27 Jahre alt" -> "Als Person in den Zwanzigern"
    t = re.sub(r"\b(\d{2}) Jahre alt\b", lambda m: _decade(m[1], "de"), t)
    t = re.sub(rf"\baus (?:der |dem |den )?({_COUNTRY})\b", lambda m: "aus " + REGION_PHRASES[_region(m[1])][2], t)
    t = RE_COUNTRY.sub(lambda m: REGION_PHRASES[_region(m.group())][3], t)
    t = re.sub(rf"\b(?:im|in dem) Studiengang ({_PROG})(?!\w)", lambda m: "in einem " + DEGREE_DE[_degree(m[1])], t)
    t = re.sub(rf"\bStudiengang ({_PROG})(?!\w)", lambda m: DEGREE_DE[_degree(m[1])], t)
    t = RE_PROG.sub(lambda m: "ein " + DEGREE_DE[_degree(m.group())], t)
    return t


def generalise_indirect(text: str, lang: str) -> str:
    """Coarsen the self-description in a (Presidio-masked) ticket; other texts stay unchanged."""
    if not has_indirect(text):
        return text
    return _generalise_de(text) if lang == "de" else _generalise_en(text)


# --- metric -------------------------------------------------------------------------------------
def _learned(text: str) -> dict[str, set]:
    """QI values an attacker can still read from the text (each key: allowed values)."""
    out: dict[str, set] = {}
    countries = {COUNTRY_ALIASES[m] for m in RE_COUNTRY.findall(text)}
    regions = {_REGION_TEXT[m] for m in RE_REGION.findall(text)}
    nats = countries | {c for c, r in C.REGION.items() if r in regions}
    if RE_ABROAD.search(text):
        nats |= {c for c in C.REGION if c != "Germany"}
    if nats:
        out["nationality"] = nats
    ages = {int(a or b) for a, b in RE_AGE.findall(text)}
    for a, b in RE_DECADE.findall(text):
        d = _DECADE_WORDS[(a or b).lower()]
        ages |= set(range(18, 20)) if d == 1 else set(range(10 * d, 10 * d + 10))
    if ages:
        out["age"] = ages
    progs = {PROGRAMME_ALIASES[m] for m in RE_PROG.findall(text)}
    degrees = set(RE_DEGREE.findall(text))
    progs |= {c for c, v in C.PROGRAMMES.items() if v[1] in degrees}
    if progs:
        out["programme"] = progs
    genders = {GENDER_WORDS[next(g for g in m.groups() if g).lower()] for m in RE_GENDER.finditer(text)}
    if genders:
        out["gender"] = genders
    return out


def anonymity_set_sizes(texts: list[str], students: pd.DataFrame) -> list[int | None]:
    """Students matching everything the text reveals about its writer; None without a self-description."""
    cols = {c: students[c].to_numpy() for c in ("nationality", "age", "programme", "gender")}
    sizes: list[int | None] = []
    for text in texts:
        if not has_indirect(text):
            sizes.append(None)
            continue
        mask = np.ones(len(students), dtype=bool)
        for col, values in _learned(text).items():
            mask &= np.isin(cols[col], list(values))
        sizes.append(int(mask.sum()))
    return sizes


# --- LLM rewrite --------------------------------------------------------------------------------
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_BATCH, LLM_MAX_NEW_TOKENS = 8, 160
LLM_SYSTEM = (
    "You rewrite student helpdesk messages for privacy. Rewrite the message so that no individual "
    "student can be singled out: remove or make vague any age, gender, country of origin, "
    "nationality, study programme name or code, and phrases like 'the only' or 'die einzige'. "
    "Keep the complaint or request and its details, and keep placeholders such as <PERSON> or "
    "<LOCATION> unchanged. The message is in {language}; answer in {language}. "
    "Output only the rewritten message."
)
LANGUAGE_NAME = {"en": "English", "de": "German"}
LLM_EXAMPLES = [
    ("I am a 31-year-old male student from Kenya in MRO, and the lab booking system rejects my login.",
     "I am a student, and the lab booking system rejects my login."),
    ("Als einzige Person aus <LOCATION> im Studiengang MDM finde ich die Sprechzeiten zu kurz.",
     "Als Studierende finde ich die Sprechzeiten zu kurz."),
]


@functools.cache  # loading the model dominates runtime; once per process
def _load_llm(model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)  # bf16 matmuls are slow on most CPUs
    lm.eval()
    return tok, lm


def llm_rewrite(texts: list[str], langs: list[str], model: str = LLM_MODEL) -> list[str]:
    """Rewrite texts with a self-description by a local instruct model (greedy, so reproducible)."""
    import torch

    if len(texts) != len(langs):
        raise ValueError("texts and langs differ in length")
    todo = [i for i, t in enumerate(texts) if has_indirect(t)]
    out = list(texts)
    if not todo:
        return out
    tok, lm = _load_llm(model)
    shots = [m for user, bot in LLM_EXAMPLES for m in ({"role": "user", "content": user},
                                                        {"role": "assistant", "content": bot})]
    for b in range(0, len(todo), LLM_BATCH):
        idx = todo[b:b + LLM_BATCH]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": LLM_SYSTEM.format(language=LANGUAGE_NAME[langs[i]])}, *shots,
             {"role": "user", "content": texts[i]}],
            tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True)
        with torch.inference_mode():
            gen = lm.generate(**enc, max_new_tokens=LLM_MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for i, seq in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[i] = tok.decode(seq, skip_special_tokens=True).strip()
    return out
