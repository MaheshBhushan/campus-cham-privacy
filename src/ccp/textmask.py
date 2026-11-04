"""Free-text PII detection and masking with Microsoft Presidio (NER + patterns), German and English.

Detection engines (combinable):
- spacy:   spaCy NER (en_core_web_md / de_core_news_md). Misses lowercase and non-German/English names.
- gliner:  GLiNER multilingual PII model, a small transformer that is prompted with label names
           ("person", "address" ...) instead of a fixed tag set, so it does not depend on capitals.
- roster:  the university knows who it enrols and employs, so names from the student and staff
           register become a case-insensitive deny-list (a lookup, no statistics involved).
Format-based identifiers (email, phone, IBAN, matriculation number, street address) always use patterns.

Two modes:
- mask:       "<PERSON>", "<EMAIL_ADDRESS>" ... (anonymisation of the text; nothing to link back)
- pseudonym:  keyed HMAC tokens, e.g. "PERSON_3f9a1c0b". The same person gets the same token in
              every ticket, so analysts can still count repeat contacts. Whoever holds the key can
              re-link, so per GDPR Art. 4(5) this is pseudonymisation, not anonymisation: the key
              stays with the data owner.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import re

import pandas as pd
from faker.providers.lorem.de_DE import Provider as LoremDe
from faker.providers.lorem.en_US import Provider as LoremEn
from gliner import GLiNER
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine, NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    DateRecognizer,
    EmailRecognizer,
    GLiNERRecognizer,
    IbanRecognizer,
    IpRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from spacy.lang.de.stop_words import STOP_WORDS as STOP_DE
from spacy.lang.en.stop_words import STOP_WORDS as STOP_EN

LANGS = ("en", "de")
NER_ENGINES = ("spacy", "gliner", "spacy+gliner")
ENTITIES = [
    "PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "DATE_TIME",
    "MATRICULATION_NO", "CREDIT_CARD", "IP_ADDRESS", "NRP",
]

NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_md"},
        {"lang_code": "de", "model_name": "de_core_news_md"},
    ],
    "ner_model_configuration": {
        # en uses OntoNotes labels (PERSON, GPE, NORP...), de uses WikiNER (PER, LOC, ORG, MISC).
        "model_to_presidio_entity_mapping": {
            "PER": "PERSON", "PERSON": "PERSON", "NORP": "NRP", "FAC": "LOCATION", "LOC": "LOCATION",
            "GPE": "LOCATION", "DATE": "DATE_TIME", "TIME": "DATE_TIME",
        },
        "low_confidence_score_multiplier": 0.4,
        "low_score_entity_names": [],
        "labels_to_ignore": ["ORG", "ORGANIZATION", "MISC", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"],
    },
}

# urchade/gliner_multi_pii-v1 fine-tuned on synthetic multilingual PII (en, de, fr, ...); mDeBERTa
# encoder, ~1.2 GB, CPU is fine. Unlike the urchade original it ships a ready tokenizer.json, which
# transformers 5 needs (converting mDeBERTa's spm.model would require protobuf).
GLINER_MODEL = "E3-JSI/gliner-multi-pii-domains-v1"
# GLiNER label prompt -> Presidio entity. Formats with exact patterns (IBAN, matriculation no.) are
# left to the patterns; every extra label adds prompt length and runtime.
GLINER_LABELS = {
    "person": "PERSON", "address": "LOCATION", "phone number": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS", "date of birth": "DATE_TIME",
}

# Register tokens that are everyday words would mask ordinary text ("auch", "will", "rose").
# Lexicon: spaCy's English/German stop words plus Faker's common-word lists (~970 en, ~500 de).
COMMON_WORDS = frozenset(w.lower() for w in (*STOP_EN, *STOP_DE, *LoremEn.word_list, *LoremDe.word_list))
TITLES = frozenset({"prof", "dr", "frau", "herr", "mr", "mrs", "ms"})


def _custom_recognizers(lang: str) -> list[PatternRecognizer]:
    ctx = {
        "en": ["matriculation", "student", "id", "number", "no"],
        "de": ["matrikelnummer", "matr", "nr", "matrikel", "student"],
    }[lang]
    return [
        # THD matriculation numbers: 8 digits (format is an assumption of the fake data). Scored above
        # spaCy's 0.85 when context words are near, so the span is typed as a matriculation number, not a date.
        PatternRecognizer(
            supported_entity="MATRICULATION_NO",
            supported_language=lang,
            patterns=[Pattern("matno_8_digits", r"\b[12]\d{7}\b", 0.6)],
            context=ctx,
        ),
        # German mobile numbers in common spellings: +49 151 1234567, 0151/1234567, +491511234567.
        # The generic phone recogniser scores these at 0.4 or misses the unspaced +49 form.
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            supported_language=lang,
            patterns=[Pattern("de_mobile", r"(?<![\w+])(?:\+49\s?|0)1[5-7]\d(?:[\s/]?\d){7,8}\b", 0.9)],
        ),
        # German street address: "Kuhlweg 78, 93413 Cham". spaCy rarely tags street names.
        # Presidio compiles patterns with IGNORECASE, so the capitalisation check is scoped with (?-i:).
        PatternRecognizer(
            supported_entity="LOCATION",
            supported_language=lang,
            patterns=[Pattern(
                "de_street_address",
                r"(?-i:\b[A-ZÄÖÜ][\wäöüß.-]*(?:\s[A-ZÄÖÜ][\wäöüß.-]*)*\s\d{1,4}[a-z]?,?\s\d{5}\s[A-ZÄÖÜ][\wäöüß]+(?:\s(?:im|in|an|der|bei)\s[A-ZÄÖÜ][\wäöüß]+)*)",
                0.7,
            )],
        ),
    ]


def _roster_terms(roster: tuple[str, ...]) -> list[str]:
    """Register entries -> deny-list terms, longest first so the regex prefers the full name.

    A multi-word entry ("Ana Julia", "Prof. Dr. Wiek", "Hans-Georg") is kept whole: as an exact
    phrase it is not ordinary text. Its single words are kept only if they have at least 3 letters and
    are not a common English/German word or a title. 1-2 letter tokens ("Li", "Om", "da", "II") collide
    with abbreviations, units, particles and Roman numerals in a whole-word, case-insensitive match;
    common words would mask everyday text. Those names are still caught inside a full name or by NER.
    """
    skip = COMMON_WORDS | TITLES
    terms = set()
    for entry in roster:
        entry = " ".join(entry.split())
        words = re.split(r"[\s-]+", entry)
        if len(words) > 1:
            terms.add(entry)
        terms.update(w for w in words if len(w) >= 3 and w.lower().rstrip(".") not in skip)
    return sorted(terms, key=lambda t: (-len(t), t))


def _roster_recognizer(roster: tuple[str, ...], lang: str) -> PatternRecognizer:
    alt = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in _roster_terms(roster))
    # Whole words, and runs of register words ("maanav banik") become one span, so a pseudonym
    # covers the full name. Not inside an email/user name ("maanav.banik@..."): the email
    # recogniser owns that span. Presidio compiles with IGNORECASE, which covers "maanav banik".
    regex = rf"(?<![\w@])(?<!\w\.)(?:{alt})(?:\s+(?:{alt}))*(?![\w@])(?!\.\w)"
    return PatternRecognizer(
        supported_entity="PERSON", supported_language=lang, name="RosterRecognizer",
        patterns=[Pattern("student_staff_register", regex, 0.95)],
    )


@functools.cache  # one model in memory, shared by the en and de recognisers
def _gliner_model(name: str) -> GLiNER:
    return GLiNER.from_pretrained(name, map_location="cpu")


class SharedGLiNERRecognizer(GLiNERRecognizer):
    """Presidio's GLiNER recogniser with a process-wide model, prompted only with its own labels
    (the stock version also sends every other requested entity name, e.g. "IBAN_CODE", as a label)."""

    def load(self) -> None:
        self.gliner = _gliner_model(self.model_name)

    def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None) -> list[RecognizerResult]:
        return super().analyze(text, [e for e in entities if e in self.supported_entities], nlp_artifacts)


@functools.cache  # loading both spaCy models dominates start-up; shared by all analyzer variants
def _nlp_engine() -> NlpEngine:
    return NlpEngineProvider(nlp_configuration=NLP_CONFIG).create_engine()


@functools.cache
def build_analyzer(ner: str = "spacy", roster: tuple[str, ...] | None = None) -> AnalyzerEngine:
    if ner not in NER_ENGINES:
        raise ValueError(f"ner must be one of {NER_ENGINES}, got {ner!r}")
    registry = RecognizerRegistry(supported_languages=list(LANGS))
    for lang in LANGS:
        if "spacy" in ner:
            registry.add_recognizer(SpacyRecognizer(supported_language=lang, supported_entities=["PERSON", "LOCATION", "NRP", "DATE_TIME"]))
        if "gliner" in ner:
            registry.add_recognizer(SharedGLiNERRecognizer(
                supported_language=lang, entity_mapping=GLINER_LABELS, model_name=GLINER_MODEL,
                threshold=0.4, map_location="cpu",
            ))
        if roster:
            registry.add_recognizer(_roster_recognizer(roster, lang))
        registry.add_recognizer(EmailRecognizer(supported_language=lang))
        registry.add_recognizer(PhoneRecognizer(supported_language=lang, supported_regions=("DE", "AT", "CZ", "GB", "US", "IN")))
        registry.add_recognizer(IbanRecognizer(supported_language=lang))
        registry.add_recognizer(DateRecognizer(supported_language=lang))
        registry.add_recognizer(CreditCardRecognizer(supported_language=lang))
        registry.add_recognizer(IpRecognizer(supported_language=lang))
        for r in _custom_recognizers(lang):
            registry.add_recognizer(r)
    return AnalyzerEngine(nlp_engine=_nlp_engine(), registry=registry, supported_languages=list(LANGS))


def _operators(mode: str, key: bytes) -> dict[str, OperatorConfig]:
    if mode == "mask":
        return {"DEFAULT": OperatorConfig("replace")}  # "<ENTITY_TYPE>"
    ops = {}
    for ent in ENTITIES:
        def token(text: str, ent: str = ent) -> str:
            digest = hmac.new(key, text.strip().lower().encode(), hashlib.sha256).hexdigest()[:8]
            return f"{ent}_{digest}"
        ops[ent] = OperatorConfig("custom", {"lambda": token})
    return ops


def anonymise_tickets(
    tickets: pd.DataFrame,
    mode: str = "mask",
    key: bytes = b"",
    score_threshold: float = 0.35,
    roster: list[str] | None = None,
    ner: str = "spacy",
) -> pd.DataFrame:
    """roster: names from the student/staff register (first names, last names, staff); None = no roster.
    ner: "spacy", "gliner" or "spacy+gliner"."""
    if mode == "pseudonym" and not key:
        raise ValueError("pseudonym mode needs a secret key")
    analyzer = build_analyzer(ner, tuple(sorted(set(roster))) if roster else None)
    anonymizer = AnonymizerEngine()
    ops = _operators(mode, key)
    detected, cleaned = [], []
    for text, lang in zip(tickets["text"], tickets["lang"]):
        res = analyzer.analyze(text=text, language=lang, entities=ENTITIES, score_threshold=score_threshold)
        detected.append(json.dumps([
            {"start": r.start, "end": r.end, "label": r.entity_type, "score": round(r.score, 2)} for r in res
        ]))
        cleaned.append(anonymizer.anonymize(text=text, analyzer_results=res, operators=ops).text)
    out = tickets.drop(columns=["matriculation_no"], errors="ignore").copy()
    out["detected"] = detected
    out["text_anonymised"] = cleaned
    return out
