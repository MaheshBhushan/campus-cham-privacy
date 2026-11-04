"""Free-text PII detection and masking with Microsoft Presidio (spaCy NER + patterns), German and English.

Two modes:
- mask:       "<PERSON>", "<EMAIL_ADDRESS>" ... (anonymisation of the text; nothing to link back)
- pseudonym:  keyed HMAC tokens, e.g. "PERSON_3f9a1c0b". The same person gets the same token in
              every ticket, so analysts can still count repeat contacts. Whoever holds the key can
              re-link, so per GDPR Art. 4(5) this is pseudonymisation, not anonymisation: the key
              stays with the data owner.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import functools

import pandas as pd
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    DateRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    IpRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

LANGS = ("en", "de")
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


@functools.cache  # loading both spaCy models dominates runtime; build once per process
def build_analyzer() -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration=NLP_CONFIG).create_engine()
    registry = RecognizerRegistry(supported_languages=list(LANGS))
    for lang in LANGS:
        registry.add_recognizer(SpacyRecognizer(supported_language=lang, supported_entities=["PERSON", "LOCATION", "NRP", "DATE_TIME"]))
        registry.add_recognizer(EmailRecognizer(supported_language=lang))
        registry.add_recognizer(PhoneRecognizer(supported_language=lang, supported_regions=("DE", "AT", "CZ", "GB", "US", "IN")))
        registry.add_recognizer(IbanRecognizer(supported_language=lang))
        registry.add_recognizer(DateRecognizer(supported_language=lang))
        registry.add_recognizer(CreditCardRecognizer(supported_language=lang))
        registry.add_recognizer(IpRecognizer(supported_language=lang))
        for r in _custom_recognizers(lang):
            registry.add_recognizer(r)
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=list(LANGS))


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
    tickets: pd.DataFrame, mode: str = "mask", key: bytes = b"", score_threshold: float = 0.35
) -> pd.DataFrame:
    if mode == "pseudonym" and not key:
        raise ValueError("pseudonym mode needs a secret key")
    analyzer = build_analyzer()
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
