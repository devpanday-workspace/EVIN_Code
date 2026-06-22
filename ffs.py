"""
GDELT Semantic Sphere Pipeline
================================
Decomposes each (subject, action, object) triplet into sphere-specific scores
WITHOUT a raw LLM call. Spheres are rule-based first (CAMEO QuadClass /
Goldstein / action keywords); a neural net is only an optional, swappable
fallback inside entity resolution.

Pipeline (see accompanying diagram):
triplet -> ActorTypeClassifier -> CameoSphereTagger -> SphereOrchestrator
            (gates which spheres apply)      -> [Economic, ForeignTension,
                                                    InternalTension, MarketProxy]
            -> Aggregator -> Postgres

Designed to plug in downstream of gdelt_actor_chains.py's extract_triples().
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =====================================================================
# 0. VERIFIED GDELT 2.0 EVENT SCHEMA (cross-checked against the official
#    codebook + two independent published column lists — replace whatever
#    column dict you're currently using with this one)
# =====================================================================
VERIFIED_EVENT_COLUMNS = {
    1: "GlobalEventID", 2: "Day", 3: "MonthYear", 4: "Year", 5: "FractionDate",
    6: "Actor1Code", 7: "Actor1Name", 8: "Actor1CountryCode",
    9: "Actor1KnownGroupCode", 10: "Actor1EthnicCode",
    11: "Actor1Religion1Code", 12: "Actor1Religion2Code",
    13: "Actor1Type1Code", 14: "Actor1Type2Code", 15: "Actor1Type3Code",
    16: "Actor2Code", 17: "Actor2Name", 18: "Actor2CountryCode",
    19: "Actor2KnownGroupCode", 20: "Actor2EthnicCode",
    21: "Actor2Religion1Code", 22: "Actor2Religion2Code",
    23: "Actor2Type1Code", 24: "Actor2Type2Code", 25: "Actor2Type3Code",
    26: "IsRootEvent", 27: "EventCode", 28: "EventBaseCode", 29: "EventRootCode",
    30: "QuadClass", 31: "GoldsteinScale", 32: "NumMentions", 33: "NumSources",
    34: "NumArticles", 35: "AvgTone",
    36: "Actor1Geo_Type", 37: "Actor1Geo_FullName", 38: "Actor1Geo_CountryCode",
    44: "Actor2Geo_Type", 45: "Actor2Geo_FullName", 46: "Actor2Geo_CountryCode",
    60: "DATEADDED", 61: "SOURCEURL",
}
# Subset actually needed for this pipeline (1-indexed -> use cols-1 for pandas)
SPHERE_RELEVANT_COLUMNS = {
    k: v for k, v in VERIFIED_EVENT_COLUMNS.items()
    if v in {
        "Day", "Actor1Name", "Actor1CountryCode", "Actor1Type1Code",
        "Actor2Name", "Actor2CountryCode", "Actor2Type1Code",
        "EventCode", "EventRootCode", "QuadClass", "GoldsteinScale",
        "NumMentions", "SOURCEURL",
    }
}


# =====================================================================
# 1. ACTOR PROFILING — rule-based, no NN
# =====================================================================
class ActorType(Enum):
    STATE = "state"                       # sovereign country actor
    SUBNATIONAL_POLITICAL = "subnational"  # GOV/OPP/REB/MIL/etc within a country
    CORPORATE = "corporate"               # company / multinational
    INDIVIDUAL = "individual"             # named person, no clear type code
    GENERIC_ROLE = "generic_role"         # "COMPANY", "FARMER", "PRIME MINISTER"...
    UNRESOLVED = "unresolved"             # not enough signal to classify

# From CAMEO.type.txt (GDELT's actor-role lookup) — verify the full taxonomy
# against that file before relying on this in production; this is a working
# subset, not the complete official list.
DOMESTIC_POLITICAL_TYPES = {"GOV", "MIL", "OPP", "REB", "SEP", "INS", "COP",
                            "JUD", "LEG", "PTY", "ELI", "LAB"}
CORPORATE_TYPES = {"BUS", "MNC"}

# Generic role nouns GDELT falls back to when it can't resolve a specific
# entity (see prior turn) — these get GENERIC_ROLE regardless of type code.
GENERIC_ROLE_NOUNS = {
    "COMPANY", "GOVERNMENT", "MILITARY", "POLICE", "CITIZEN", "BUSINESS",
    "MEDIA", "PRESIDENT", "PRIME MINISTER", "MINISTER", "FARMER",
}


@dataclass
class ActorProfile:
    raw_name: str
    actor_code: Optional[str]
    country_code: Optional[str]
    geo_country_code: Optional[str]
    type_codes: tuple  # up to 3 CAMEO type codes, may be empty
    actor_type: ActorType = ActorType.UNRESOLVED
    confidence: float = 0.0
    market_ticker: Optional[str] = None      # filled by EntityResolver
    market_index_proxy: Optional[str] = None  # filled by EntityResolver
    reasoning: tuple = ()  # which rules fired, for debugging/audit


class ActorTypeClassifier:
    """Rule-based actor classification. No NN. ~O(1) per actor."""

    def classify(self, name: str, actor_code: Optional[str],
                country_code: Optional[str], geo_country_code: Optional[str],
                type_codes: tuple) -> ActorProfile:
        name = (name or "").strip().upper()
        types = {t for t in type_codes if t}
        reasons = []

        if name in GENERIC_ROLE_NOUNS:
            reasons.append("name_in_generic_role_set")
            return ActorProfile(name, actor_code, country_code, geo_country_code,
                                type_codes, ActorType.GENERIC_ROLE, 0.9, reasoning=tuple(reasons))

        if types & DOMESTIC_POLITICAL_TYPES:
            reasons.append(f"type_code_match:{types & DOMESTIC_POLITICAL_TYPES}")
            return ActorProfile(name, actor_code, country_code, geo_country_code,
                                type_codes, ActorType.SUBNATIONAL_POLITICAL, 0.85, reasoning=tuple(reasons))

        if types & CORPORATE_TYPES:
            reasons.append(f"type_code_match:{types & CORPORATE_TYPES}")
            return ActorProfile(name, actor_code, country_code, geo_country_code,
                                type_codes, ActorType.CORPORATE, 0.85, reasoning=tuple(reasons))

        # No type code + country_code equals the actor's own geo match + name
        # reads like a country -> GDELT matched this purely geographically.
        # if not types and country_code and country_code == geo_country_code:
        #     reasons.append("pure_geo_match_no_type_code")
        #     return ActorProfile(name, actor_code, country_code, geo_country_code,
        #                         type_codes, ActorType.STATE, 0.75, reasoning=tuple(reasons))
        if not types and country_code:
            reasons.append("country_code_no_type_code")
            return ActorProfile(
                name, actor_code, country_code, geo_country_code,type_codes, ActorType.STATE, 0.75,reasoning=tuple(reasons))

        if not types and name and " " in name and country_code:
            # weak heuristic: multi-word name, no role code, has a country
            # context -> plausibly a named individual. Low confidence by design.
            reasons.append("weak_individual_heuristic")
            return ActorProfile(name, actor_code, country_code, geo_country_code,
                                type_codes, ActorType.INDIVIDUAL, 0.4, reasoning=tuple(reasons))

        reasons.append("no_rule_matched")
        return ActorProfile(name, actor_code, country_code, geo_country_code,
                            type_codes, ActorType.UNRESOLVED, 0.0, reasoning=tuple(reasons))


# =====================================================================
# 2. ENTITY RESOLUTION — rule-based primary, NN fallback OPTIONAL (off by default)
# =====================================================================
class EntityResolver(ABC):
    @abstractmethod
    def resolve_market_instrument(self, profile: ActorProfile) -> ActorProfile:
        """Fill in market_ticker (corporate) or market_index_proxy (state)."""


class RuleBasedResolver(EntityResolver):
    """
    Static alias table lookup. Seed COMPANY_ALIASES from a maintained source
    (SEC EDGAR ticker list, Wikidata dump) rather than hand-typing it — this
    is a stub with a couple of entries to show the shape.
    """
    COMPANY_ALIASES = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "TESLA": "TSLA",
    }
    # Sovereign proxy: country -> a broad equity index/ETF ticker used as a
    # market-reaction stand-in for that country's economic exposure.
    COUNTRY_INDEX_PROXIES = {
        "USA": "^GSPC", "CHN": "MCHI", "IND": "INDA", "GBR": "EWU",
    }

    def resolve_market_instrument(self, profile: ActorProfile) -> ActorProfile:
        if profile.actor_type == ActorType.CORPORATE:
            ticker = self.COMPANY_ALIASES.get(profile.raw_name)
            if ticker:
                profile.market_ticker = ticker
        elif profile.actor_type == ActorType.STATE and profile.country_code:
            proxy = self.COUNTRY_INDEX_PROXIES.get(profile.country_code)
            if proxy:
                profile.market_index_proxy = proxy
        return profile


class EmbeddingFallbackResolver(EntityResolver):
    """
    OPTIONAL. Only invoked when RuleBasedResolver fails AND the actor type
    is CORPORATE/INDIVIDUAL (i.e. resolution is plausible but the alias
    table missed it). Requires `pip install sentence-transformers` — kept
    as a stub so the default pipeline has zero NN dependency.
    """
    def __init__(self, candidate_names: dict):
        self.candidate_names = candidate_names  # {canonical_name: ticker}
        self._model = None  # lazy-loaded only if this class is actually used

    def resolve_market_instrument(self, profile: ActorProfile) -> ActorProfile:
        raise NotImplementedError(
            "Load a small sentence-transformers model here only if/when "
            "RuleBasedResolver's hit rate proves insufficient in practice."
        )


# =====================================================================
# 3. CAMEO ACTION TAGGING — rule-based (QuadClass authoritative; keyword
#    match on the CAMEO label text for sub-domain tagging)
# =====================================================================
ECONOMIC_KEYWORDS = ("econom", "trade", "aid", "sanction", "embargo",
                    "tariff", "boycott", "fund", "invest", "financ", "business")
MILITARY_KEYWORDS = ("troops", "military", "armed forces", "blockade",
                    "attack", "bomb", "missile", "weapon", "fight")
DIPLOMATIC_KEYWORDS = ("diplomat", "meet", "summit", "treaty",
                    "negotiat", "consult", "talks")


class CameoSphereTagger:
    def __init__(self, cameo_lookup: Optional[dict] = None):
        self.cameo_lookup = cameo_lookup or {}

    def tag(self, event_code: str, quad_class: Optional[int]) -> set:
        tags = set()
        label = self.cameo_lookup.get(str(event_code), "").lower()

        if quad_class in (1, 2):
            tags.add("COOPERATION")
        elif quad_class in (3, 4):
            tags.add("CONFLICT")

        if any(kw in label for kw in ECONOMIC_KEYWORDS):
            tags.add("ECONOMIC")
        if any(kw in label for kw in MILITARY_KEYWORDS):
            tags.add("MILITARY")
        if any(kw in label for kw in DIPLOMATIC_KEYWORDS):
            tags.add("DIPLOMATIC")
        return tags


# =====================================================================
# 4. ABSTRACTED EVENT (decouples spheres from GDELT specifically, so the
#    same spheres later accept social-media-derived interactions)
# =====================================================================
@dataclass
class InteractionEvent:
    subject: ActorProfile
    object: ActorProfile
    goldstein: Optional[float]
    quad_class: Optional[int]
    num_mentions: int
    sphere_tags: set
    day: str
    source: str


@dataclass
class SphereResult:
    sphere_name: str
    applicable: bool
    score: Optional[float] = None
    confidence: float = 0.0
    features: dict = field(default_factory=dict)
    used_nn: bool = False
    module_version: str = "0.1"


# =====================================================================
# 5. SPHERE ANALYZERS — each spheres declares its own applicability
# =====================================================================
class SphereAnalyzer(ABC):
    name: str

    @abstractmethod
    def is_applicable(self, event: InteractionEvent) -> bool: ...

    @abstractmethod
    def analyze(self, event: InteractionEvent) -> SphereResult: ...


class EconomicEventsSphere(SphereAnalyzer):
    name = "economic_events"

    def is_applicable(self, event: InteractionEvent) -> bool:
        return "ECONOMIC" in event.sphere_tags

    def analyze(self, event: InteractionEvent) -> SphereResult:
        confidence = min(1.0, math.log1p(event.num_mentions) / math.log1p(50))
        return SphereResult(
            self.name, True, score=event.goldstein, confidence=confidence,
            features={"quad_class": event.quad_class, "tags": list(event.sphere_tags)},
        )


class ForeignPoliticalTensionSphere(SphereAnalyzer):
    name = "foreign_political_tension"

    def is_applicable(self, event: InteractionEvent) -> bool:
        s, o = event.subject, event.object
        return (
            s.actor_type == ActorType.STATE and o.actor_type == ActorType.STATE
            and s.country_code and o.country_code
            and s.country_code != o.country_code
        )

    def analyze(self, event: InteractionEvent) -> SphereResult:
        # Goldstein IS the dyadic tension/cooperation score; QuadClass 3/4
        # (conflict) gets a confidence boost since the action itself signals
        # tension regardless of magnitude.
        confidence = 0.6 + (0.3 if event.quad_class in (3, 4) else 0.0)
        return SphereResult(
            self.name, True, score=event.goldstein, confidence=min(confidence, 1.0),
            features={"actor1_country": event.subject.country_code,
                    "actor2_country": event.object.country_code},
        )


class InternalPoliticalTensionSphere(SphereAnalyzer):
    name = "internal_political_tension"

    def is_applicable(self, event: InteractionEvent) -> bool:
        s, o = event.subject, event.object
        same_country = s.country_code and s.country_code == o.country_code
        has_domestic_role = (
            s.actor_type == ActorType.SUBNATIONAL_POLITICAL
            or o.actor_type == ActorType.SUBNATIONAL_POLITICAL
        )
        return bool(same_country and has_domestic_role)

    def analyze(self, event: InteractionEvent) -> SphereResult:
        confidence = 0.6 + (0.2 if event.quad_class in (3, 4) else 0.0)
        return SphereResult(
            self.name, True, score=event.goldstein, confidence=min(confidence, 1.0),
            features={"country": event.subject.country_code},
        )


class MarketProxySphere(SphereAnalyzer):
    """
    Score = abnormal return in a short window around the event date,
    relative to a benchmark. Requires a MarketDataProvider plugged in —
    this pipeline doesn't have network access to price data, so it's a
    clean injection point for whatever you already have in Postgres.
    """
    name = "market_proxy"

    def __init__(self, market_data_provider=None):
        self.provider = market_data_provider  # must implement get_returns(symbol, date)

    def is_applicable(self, event: InteractionEvent) -> bool:
        return bool(
            event.subject.market_ticker or event.subject.market_index_proxy
            or event.object.market_ticker or event.object.market_index_proxy
        )

    def analyze(self, event: InteractionEvent) -> SphereResult:
        if self.provider is None:
            return SphereResult(
                self.name, True, score=None, confidence=0.0,
                features={"note": "no MarketDataProvider configured — wire one in"},
            )
        # TODO: pull window of returns around event.day for whichever
        # symbol(s) resolved, compute abnormal return vs benchmark.
        return SphereResult(self.name, True, score=None, confidence=0.0)


# =====================================================================
# 6. ORCHESTRATOR
# =====================================================================
class SphereOrchestrator:
    def __init__(self, spheres: list[SphereAnalyzer]):
        self.spheres = spheres

    def run(self, event: InteractionEvent) -> dict:
        results = {}
        for sphere in self.spheres:
            if sphere.is_applicable(event):
                results[sphere.name] = sphere.analyze(event)
            else:
                results[sphere.name] = SphereResult(sphere.name, applicable=False)
        return results


# =====================================================================
# 7. BONUS: rolling escalation/anomaly index (cheap, no NN)
# =====================================================================
class RollingIndexAggregator:
    """
    Tracks each actor's running Goldstein mean/std and flags events that
    deviate sharply from that actor's own baseline -- 'unusually tense for
    THEM', not just a low absolute score.
    """
    def __init__(self):
        self._stats: dict[str, dict] = {}  # actor_name -> {n, mean, m2}

    def update_and_score(self, actor_name: str, goldstein: float) -> float:
        s = self._stats.setdefault(actor_name, {"n": 0, "mean": 0.0, "m2": 0.0})
        s["n"] += 1
        delta = goldstein - s["mean"]
        s["mean"] += delta / s["n"]
        s["m2"] += delta * (goldstein - s["mean"])
        if s["n"] < 5:
            return 0.0  # not enough history yet
        std = math.sqrt(s["m2"] / (s["n"] - 1)) or 1e-6
        return (goldstein - s["mean"]) / std  # z-score: escalation magnitude


# =====================================================================
# 8. POSTGRES SCHEMA (reference — adapt to your existing tables)
# =====================================================================
POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS actor_registry (
    actor_id        SERIAL PRIMARY KEY,
    raw_name        TEXT NOT NULL,
    actor_type      TEXT NOT NULL,
    country_code    TEXT,
    market_ticker   TEXT,
    confidence      REAL,
    UNIQUE (raw_name, actor_type)
);

CREATE TABLE IF NOT EXISTS interaction_events (
    event_id        BIGSERIAL PRIMARY KEY,
    subject_id      INT REFERENCES actor_registry(actor_id),
    object_id       INT REFERENCES actor_registry(actor_id),
    event_code      TEXT,
    quad_class      SMALLINT,
    goldstein       REAL,
    num_mentions    INT,
    day             DATE,
    source_url      TEXT
);

CREATE TABLE IF NOT EXISTS sphere_scores (
    score_id        BIGSERIAL PRIMARY KEY,
    event_id        BIGINT REFERENCES interaction_events(event_id),
    sphere_name     TEXT NOT NULL,
    applicable      BOOLEAN NOT NULL,
    score           REAL,
    confidence      REAL,
    used_nn         BOOLEAN DEFAULT FALSE,
    module_version  TEXT,
    UNIQUE (event_id, sphere_name, module_version)
);
"""

def process_triples_dataframe(
    triples_df,
    cameo_lookup=None
):
    """
    Consumes output from app_1.extract_triples()
    and runs the full FFS semantic pipeline.
    """
    print("Lookup size:", len(cameo_lookup or {}))
    classifier = ActorTypeClassifier()

    resolver = RuleBasedResolver()

    tagger = CameoSphereTagger(
        cameo_lookup=cameo_lookup
    )

    orchestrator = SphereOrchestrator([
        EconomicEventsSphere(),
        ForeignPoliticalTensionSphere(),
        InternalPoliticalTensionSphere(),
        MarketProxySphere(
            market_data_provider=None
        ),
    ])

    rolling = RollingIndexAggregator()

    results = []

    for _, row in triples_df.iterrows():

        # --------------------------------------------------
        # SUBJECT PROFILE
        # --------------------------------------------------

        subj = classifier.classify(
            row["subject"],
            row["Actor1Code"],
            row["Actor1CountryCode"],
            row["Actor1GeoCountryCode"],
            (
                row["Actor1Type1Code"],
                row["Actor1Type2Code"],
                row["Actor1Type3Code"],
            )
        )

        # --------------------------------------------------
        # OBJECT PROFILE
        # --------------------------------------------------

        obj = classifier.classify(
            row["object"],
            row["Actor2Code"],
            row["Actor2CountryCode"],
            row["Actor2GeoCountryCode"],
            (
                row["Actor2Type1Code"],
                row["Actor2Type2Code"],
                row["Actor2Type3Code"],
            )
        )
        # if len(results) < 20:
        #     print(row["subject"],subj.actor_type,row["Actor1CountryCode"],"->",
        #         row["object"],obj.actor_type,row["Actor2CountryCode"])
        # --------------------------------------------------
        # ENTITY RESOLUTION
        # --------------------------------------------------

        subj = resolver.resolve_market_instrument(
            subj
        )

        obj = resolver.resolve_market_instrument(
            obj
        )

        # --------------------------------------------------
        # CAMEO TAGGING
        # --------------------------------------------------

        tags = tagger.tag(
            str(row["EventCode"]),
            row["QuadClass"]
        )
        if len(results) < 5:
            print(row["EventCode"], tags)
        # --------------------------------------------------
        # EVENT OBJECT
        # --------------------------------------------------

        event = InteractionEvent(
            subject=subj,
            object=obj,
            goldstein=row["GoldsteinScale"],
            quad_class=row["QuadClass"],
            num_mentions=row["NumMentions"],
            sphere_tags=tags,
            day=str(row["Day"]),
            source=row["SOURCEURL"]
        )

        # --------------------------------------------------
        # RUN SPHERES
        # --------------------------------------------------

        sphere_results = orchestrator.run(
            event
        )

        # --------------------------------------------------
        # ESCALATION INDEX
        # --------------------------------------------------

        z_score = rolling.update_and_score(
            subj.raw_name,
            row["GoldsteinScale"]
        )

        results.append({

            "subject":
                subj.raw_name,

            "subject_type":
                subj.actor_type.value,

            "object":
                obj.raw_name,

            "object_type":
                obj.actor_type.value,

            "event_code":
                row["EventCode"],

            "goldstein":
                row["GoldsteinScale"],

            "sphere_results":
                sphere_results,

            "escalation_z":
                z_score,

            "source":
                row["SOURCEURL"]
        })

    return results


