import os
import sys
import csv
import time
import json
import re
import argparse
from pathlib import Path

import pandas as pd
import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError

SCRIPT_DIR = Path(__file__).parent
# Default input/output. Both overridable via --csv and --output CLI flags.
CSV_PATH = SCRIPT_DIR / "Conquest-Input.csv"
RESULTS_PATH = SCRIPT_DIR / "Conquest-Output.csv"

MODEL = "gemini-2.5-flash"
API_PAUSE_SECONDS = 1.0

# Only attempt to download decks from these hosts. Others (OneDrive, Google
# Drive share links, etc.) are auth-gated and can't be fetched reliably.
FETCHABLE_DECK_HOSTS = ("conquestbits.org",)
MAX_DECK_BYTES = 20 * 1024 * 1024  # 20 MB cap before we skip an attachment
DECK_DOWNLOAD_TIMEOUT = 30

# Gemini's inline file parts only accept these MIME types for documents. PowerPoint
# (.pptx), Word (.docx), Keynote, etc. will be rejected with HTTP 400. Anything not
# on this list gets downloaded but NOT attached -- the eval falls back to text-only.
SUPPORTED_DECK_MIMES = {"application/pdf"}

_client = None


def get_client():
    """Lazy Gemini client init so the module can be imported without an API key
    (useful for tests, validator self-checks, and importing helpers elsewhere)."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def safe(value) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "N/A"


def build_prompt(row, deck_attached: bool) -> str:
    if deck_attached:
        deck_instruction = (
            "A pitch deck PDF is ATTACHED to this message. Use it as EVIDENCE about WHAT the "
            "company does (product, customer, GTM, traction, the founder's own TAM claim, "
            "claimed differentiation) — NOT as the source of truth. Founders systematically "
            "overstate both TAM and moat strength. Extract the deck's own claims, then produce "
            "INDEPENDENT estimates from live web research and cross-check. "
            "Set \"pitch_deck_accessed\" to true and note what you read in \"pitch_deck_notes\"."
        )
    else:
        deck_instruction = (
            "No pitch deck is attached. Set \"pitch_deck_accessed\" to false, "
            "\"pitch_deck_notes\" to \"no deck attached\", \"deck_tam_claim\" to \"not available\", "
            "and \"deck_moat_claim\" to \"not available\". "
            "This does NOT reduce the depth of analysis required — perform the same full web "
            "research as you would with a deck."
        )

    return f"""
You are a disciplined venture capital analyst performing automated due diligence.
You will produce THREE independent integer scores (1-10) for the startup below:
  A) MARKET SIZE
  B) DIFFERENTIATION / MOAT
  C) PROBLEM VALIDATION

==================================================================================
ANTI-BIAS GUARDRAILS — READ AND APPLY BEFORE SCORING
==================================================================================
You MUST adhere to all of the following. Violating any of these is a failure mode.

1. INDEPENDENCE. Score each of A, B, C separately. A high market score does NOT
   imply a high moat or problem score, and vice versa. A startup can have a huge
   market with weak product-market fit, or a tiny niche with rabid demand.
   If you notice all three scores converging to the same number, re-examine.

2. SECTOR-NEUTRAL. Hype sectors (AI, Web3, climate, gen-AI, quantum) get NO bonus.
   Boring sectors (logistics, B2B SaaS, agri, manufacturing, MSME tooling) get NO
   penalty. Score the underlying business, not the category fashion.

3. GEOGRAPHY-NEUTRAL. India-focused startups are scored on the size of THEIR
   actual addressable market — never penalized for not being global. A large
   India-only market is a large market.

4. FOUNDER-IDENTITY-NEUTRAL. For sections A, B, C, the founder's name, gender,
   ethnicity, school, or background DOES NOT FACTOR INTO THESE SCORES. Those are
   measured in a separate founder evaluation. Score the BUSINESS here.

5. AESTHETICS-NEUTRAL. Deck design quality, slide polish, photo quality — all
   irrelevant. A beautiful deck for a weak business is still a weak business.

6. BUZZWORDS = ZERO. Words like "AI-powered", "blockchain", "first-of-its-kind",
   "revolutionary", "disrupting", "Uber for X", "Web3", "next-gen" carry no
   weight unless followed by SPECIFIC technical or operational evidence.

7. FOUNDER-CLAIM SKEPTICISM. Treat all founder claims as hypotheses to verify,
   not facts. Founders systematically overstate TAM, traction, moats, and PMF.
   Always cross-check claims against independent sources.

8. CALIBRATION. Most real-world startup batches cluster at 4-6. Scores of 9-10
   should be RARE (reserved for genuinely category-defining opportunities).
   If you find yourself scoring 7+ on all three, you are probably anchoring on
   the founder's narrative — reset and re-examine the evidence.

9. NO HALLUCINATION. If you cannot find evidence for a claim via Google Search
   or in the deck, say so in the relevant field ("unable to verify") rather than
   invent supporting detail. Better to score lower with low confidence than to
   fabricate justification.

10. WHEN IN DOUBT, ROUND DOWN. If genuinely torn between two adjacent scores,
    pick the LOWER one and explain the upside case in the summary.

==================================================================================
STARTUP PROFILE
==================================================================================
- Company Name: {safe(row.get('Startup Name'))}
- Sector: {safe(row.get('Sector'))}
- Business Model: {safe(row.get('Business Model'))}
- Current Stage: {safe(row.get('Stage'))}
- Location: {safe(row.get('Location'))}
- Startup Age: {safe(row.get('Startup Age'))}
- Team Size: {safe(row.get('Team Size'))}
- Founder LinkedIn: {safe(row.get('LinkedIn'))}
- Pitch Deck URL: {safe(row.get('Pitch Deck URL'))}

FOUNDER-DECLARED FIELDS (treat as claims to verify, not facts)
- Problem Being Solved: {safe(row.get('Problem Solving'))}
- Solution Provided: {safe(row.get('Solution'))}
- Target Customer Segment: {safe(row.get('Target Customer'))}
- USPs (claimed): {safe(row.get('USPs'))}
- Achievements (claimed): {safe(row.get('Achievements'))}
- Past Accelerators: {safe(row.get('Past Accelerators'))}
- Has Patents (claimed): {safe(row.get('Has Patents'))}
- Patent Details (claimed): {safe(row.get('Patent Details'))}
- Has Revenue: {safe(row.get('Has Revenue'))}
- MRR (claimed, INR/month): {safe(row.get('MRR'))}
- Monthly Burn (INR/month): {safe(row.get('Monthly Burn'))}
- Funding Status: {safe(row.get('Funding Status'))}
- Funding From: {safe(row.get('Funding From'))}
- Grant Details: {safe(row.get('Grant Details'))}
- Demo Video Provided: {"yes" if safe(row.get('Demo Video Link')) != "N/A" else "no"}

FOUNDER-DECLARED CHALLENGES (top 3 areas the founder is worried about — useful for problem articulation and self-awareness)
- Challenge 1: {safe(row.get('Challenge 1'))} | detail: {safe(row.get('Challenge 1 Details'))}
- Challenge 2: {safe(row.get('Challenge 2'))} | detail: {safe(row.get('Challenge 2 Details'))}
- Challenge 3: {safe(row.get('Challenge 3'))} | detail: {safe(row.get('Challenge 3 Details'))}

PITCH DECK HANDLING
{deck_instruction}

==================================================================================
MANDATORY RESEARCH METHOD (do ALL of these for EVERY startup, deck or no deck)
==================================================================================
1. Use the Google Search tool. At minimum search for:
   a) The startup name itself (news, funding, coverage, existence)
   b) The sector + "market size" / "TAM" / "CAGR" with the latest year
   c) 1-3 comparable companies in the same space (for sizing AND competitive density)
   d) Regional reports if India-focused (Inc42, RedSeer, Tracxn, Bain India, YourStory)
   e) For MOAT: company name + "patent" / "IP" / "open source" to verify defensibility
   f) For PROBLEM: search terms like "<problem> survey", "<problem> spend", "<problem> statistics"

2. Form INDEPENDENT estimates. Do NOT just copy numbers/claims from the deck or
   the first search hit. State the basis of your estimate.

3. If the deck disagrees with your independent estimate, USE YOURS and explain
   the gap in the relevant summary.

4. Cite the actual search topics / sources in "web_sources_used". If you did
   NOT search, set "market_confidence" to "low" and explain.

==================================================================================
A) MARKET SIZE SCORING — multi-factor, NOT just raw dollars
==================================================================================
Weigh ALL of these, then assign one integer 1-10:
  (a) Size of the realistically addressable market for THIS startup's wedge
      (SAM, not the whole sector's TAM)
  (b) Growth rate / CAGR — a $2B market growing 30% YoY can beat a $50B stagnant one
  (c) Tailwinds (regulation, tech shift, capital inflow, demographics, behavior change)
  (d) Headwinds (entrenched incumbents, commoditization, regulatory risk, capital-intensity)
  (e) Geographic reach (India-only vs globally expandable, sized correctly either way)
  (f) Timing — is the market forming, growing, mature, or saturating?

Market rubric:
  1-2  = tiny niche or shrinking; no realistic upside (<$100M SAM, declining)
  3-4  = small / fragmented, weak growth or hostile dynamics (~$100M-$1B SAM)
  5-6  = solid mid-size market (~$1-5B SAM), moderate growth, ordinary dynamics
  7-8  = large market ($5-20B SAM) with strong growth or major tailwinds
  9-10 = massive (>$20B SAM) hyper-growth or category-defining inflection

Market calibration anchors (use these to keep yourself consistent):
  - Stripe-at-founding (global online payments): 10
  - Razorpay-at-founding (India digital payments): 8
  - A regional D2C apparel brand: 5
  - Niche B2B SaaS for one job function in one country: 4
  - A college-campus-only food delivery app: 2

==================================================================================
B) DIFFERENTIATION / MOAT SCORING
==================================================================================
Assess how defensible this startup actually is against well-funded competition.
Identify which (if any) of these moat TYPES are genuinely present — be skeptical:
  - Tech / IP moat: GRANTED patents (verify), proprietary algorithms, hard-to-replicate engineering
  - Data moat: proprietary dataset compounding with use; network effect on data
  - Network effects: more users -> more value for other users (genuine two-sided dynamic)
  - Brand / community: defensible mindshare or strong organic community
  - Switching costs / lock-in: integrations, workflows, data trapped in product
  - Regulatory moat: licenses, certifications, regulated activity others can't enter easily
  - Distribution moat: exclusive channels, deep partnerships, embedded in customer ops
  - Scale / cost: structural cost advantage from scale

For EACH claimed moat, ask: is there EVIDENCE, or is it just a USP slide?
  - "Filed patents" != "granted patents"
  - "AI-powered" != tech moat
  - "Marketplace" != network effects
  - "Partnership announced" != exclusive distribution
  - "Proprietary algorithm" without specifics = no moat

Moat rubric:
  1-2  = commodity offering in a crowded space; no real differentiation; trivially replicable
  3-4  = some differentiation but easily copied; weak/temporary moats; feature-level diff
  5-6  = ONE real but vulnerable moat (e.g. brand, modest data, partnerships); a
         well-funded entrant could erode it within 2 years
  7-8  = TWO+ meaningful moats compounding (e.g. tech IP + distribution; data + switching costs)
  9-10 = step-change defensibility incumbents cannot easily replicate (genuine network
         effects at scale, granted patents in active use, structural data lead, regulatory monopoly)

Moat calibration anchors:
  - Google Search 2005 (data + scale + network effects): 10
  - AWS 2010 (switching costs + scale): 9
  - Stripe 2015 (developer brand + integrations): 7
  - A SaaS with strong brand but no patents/network: 5
  - A standard e-commerce store with no unique IP or distribution: 2

==================================================================================
C) PROBLEM VALIDATION SCORING
==================================================================================
Assess how REAL, URGENT, and VALIDATED the problem is. This is independent of
market size (a huge market can have weakly-validated problems and vice versa).

Evaluate ALL of these dimensions, then assign one integer 1-10:
  (a) Problem severity: is this a "must solve" / painkiller, or a "nice to have" / vitamin?
  (b) Problem frequency: daily / weekly / monthly / yearly / one-time?
  (c) Problem prevalence: how many people or businesses have this problem?
  (d) Existing willingness-to-pay: do users currently spend money or significant
      time on workarounds? (Existing spend is the strongest demand signal.)
  (e) Articulation quality: does the founder clearly describe WHO has this problem,
      HOW OFTEN, and WHY existing solutions fail? Vague target = bad articulation.
  (f) Evidence of demand: traction (paying users, MRR, waitlist), cited user
      research, testimonials, analogous market signals.

Problem rubric:
  1-2  = "Solution looking for a problem"; problem is hypothetical/vague; no
         evidence anyone wants this; vitamin not painkiller
  3-4  = Real but minor / nice-to-have problem; few would pay; existing tools
         already address it adequately
  5-6  = Real problem with some validation; clear target user; moderate
         willingness to pay; some traction OR analogous market signals
  7-8  = Severe, frequent, well-validated problem; clear evidence of existing
         spend on inferior solutions; founder articulates target user precisely;
         strong demand signals (e.g. paying users, growing waitlist)
  9-10 = Universal, painful, urgent problem affecting massive cohorts; clear
         existing spend; founder shows deep customer empathy + concrete validation;
         rare "obvious in hindsight" problems (think "I can't get a taxi" pre-Uber)

Problem calibration anchors:
  - "Most adults in Kenya can't access basic financial services" (M-Pesa): 10
  - "Small businesses can't accept online payments without 6-week bank setup" (Stripe): 8
  - "Teams struggle to organize tasks across email and Slack" (Asana — real but crowded): 5
  - "I want a feed of my friends' workout activity" (low pain, low willingness to pay): 3
  - "An app to remind me to drink water" (vitamin not painkiller): 2

==================================================================================
OUTPUT
==================================================================================
Return ONLY a raw JSON object (no markdown fences, no prose outside JSON), in
EXACTLY this schema. All string values MUST be single-line (no newlines inside strings).
All scores MUST be integers in [1, 10]. All confidence values MUST be one of
"low", "medium", "high".

{{
  "market_size_score": 7,
  "calculated_tam": "Your INDEPENDENT TAM, e.g. '$12B SAM by 2030 (India)' + 1-line basis",
  "deck_tam_claim": "What the deck/founder claimed, or 'not stated' / 'not available'",
  "cagr": "Your researched CAGR + year range, e.g. '18% (2024-2030)', or 'unknown'",
  "growth_tailwinds": "1-2 key tailwinds, single line",
  "growth_headwinds": "1-2 key headwinds, single line",
  "geographic_scope": "e.g. 'India-only', 'India + SEA', 'Global'",
  "market_analysis_summary": "3-5 sentences integrating evidence, explaining why this exact score and not one notch higher or lower.",
  "market_confidence": "low | medium | high",

  "moat_score": 6,
  "moat_types_present": "Comma list of moats GENUINELY present with evidence, e.g. 'Tech IP, Distribution'. Empty string if none.",
  "deck_moat_claim": "What the deck/founder claimed as their moat/USP, or 'not stated'",
  "moat_evidence": "Specific evidence for the moats listed (e.g. 'Patent US-12345 granted 2023', 'Exclusive HDFC partnership verified via press release'). If moat_score >= 7, this MUST be substantive (50+ chars).",
  "moat_risks": "What would erode the moats — 1-2 sentences",
  "moat_analysis_summary": "3-5 sentences justifying this exact score. Single line.",
  "moat_confidence": "low | medium | high",

  "problem_score": 7,
  "problem_severity": "high | medium | low",
  "problem_frequency": "daily | weekly | monthly | yearly | one-time | unclear",
  "existing_willingness_to_pay": "Brief: are people already spending on this? Workarounds? Quote a dollar amount or alternative if possible.",
  "demand_evidence": "Concrete evidence: traction numbers, testimonials, analogous market data. If problem_score >= 7, this MUST be substantive (30+ chars).",
  "problem_red_flags": "e.g. 'solution looking for a problem', 'vague TG', 'no demand evidence', or 'none observed'",
  "problem_analysis_summary": "3-5 sentences justifying this exact score. Single line.",
  "problem_confidence": "low | medium | high",

  "web_sources_used": "Short comma-separated list of search topics or source names you actually consulted",
  "pitch_deck_accessed": false,
  "pitch_deck_notes": "What you read from the deck, or 'no deck attached'. Single line."
}}
""".strip()


def is_fetchable_deck(url: str) -> bool:
    if url == "N/A":
        return False
    return any(host in url for host in FETCHABLE_DECK_HOSTS)


def download_deck(url: str):
    """Return (bytes, mime_type, status_msg) or (None, None, error_msg)."""
    try:
        resp = requests.get(
            url, timeout=DECK_DOWNLOAD_TIMEOUT, allow_redirects=True, stream=True
        )
    except requests.RequestException as e:
        return None, None, f"download failed: {e.__class__.__name__}: {e}"

    if resp.status_code != 200:
        return None, None, f"download failed: HTTP {resp.status_code}"

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    data = resp.content
    if not data:
        return None, None, "download failed: empty body"
    if len(data) > MAX_DECK_BYTES:
        return None, None, f"download skipped: file too large ({len(data)} bytes)"

    # Detect MIME: trust PDF magic bytes first, then content-type, then URL hint.
    is_pdf_magic = data[:4] == b"%PDF"
    if is_pdf_magic or "pdf" in content_type or url.lower().endswith(".pdf"):
        mime = "application/pdf"
    elif content_type:
        mime = content_type
    else:
        mime = "application/octet-stream"

    # Skip attachment for anything Gemini can't read inline (pptx, docx, etc.)
    # The row still gets fully evaluated from text + Google Search.
    if mime not in SUPPORTED_DECK_MIMES:
        size_kb = len(data) / 1024
        return None, None, (
            f"downloaded {size_kb:,.0f} KB but format '{mime}' is not Gemini-supported "
            f"-- skipping attachment (text-only eval)"
        )

    return data, mime, f"downloaded {len(data)} bytes ({mime})"


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Some responses include prose before/after the JSON object. Grab the first {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


# -------- Output validation --------------------------------------------------
# Validates structure/types/ranges of model output. Surfaces warnings to the
# CSV (column `validation_warnings_*`) so suspect rows can be audited later.
# Validation NEVER blocks writing — we always persist the row, with warnings.

ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_SEVERITY = {"high", "medium", "low"}
ALLOWED_FREQUENCY = {"daily", "weekly", "monthly", "yearly", "one-time", "unclear"}


def _coerce_score(value, field_name, errors):
    """Coerce to int in [1,10]; record an error if impossible. Returns coerced value or None."""
    if value is None:
        errors.append(f"{field_name}: missing")
        return None
    if isinstance(value, bool):
        errors.append(f"{field_name}: must be int, got bool {value!r}")
        return None
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float) and value.is_integer():
        coerced = int(value)
    elif isinstance(value, str):
        m = re.match(r"^\s*(\d+)\s*$", value)
        if not m:
            errors.append(f"{field_name}: not parseable as int, got {value!r}")
            return None
        coerced = int(m.group(1))
    else:
        errors.append(f"{field_name}: unexpected type {type(value).__name__} ({value!r})")
        return None
    if not (1 <= coerced <= 10):
        errors.append(f"{field_name}: out of range 1-10, got {coerced}")
        return coerced  # keep value for visibility even if out of range
    return coerced


def _check_enum(value, allowed, field_name, errors):
    if value is None or str(value).strip() == "":
        errors.append(f"{field_name}: missing")
        return
    v = str(value).strip().lower()
    if v not in allowed:
        errors.append(f"{field_name}: must be one of {sorted(allowed)}, got {value!r}")


def _check_substantive(value, field_name, min_chars, errors):
    """High scores must be backed by substantive supporting text."""
    text = (value or "").strip() if isinstance(value, str) else ""
    if len(text) < min_chars:
        errors.append(
            f"{field_name}: insufficient detail ({len(text)} chars, need >= {min_chars})"
        )


def validate_market_moat_problem(result: dict) -> list[str]:
    """Mutates result to coerce score types where possible; returns warning list."""
    errors: list[str] = []

    # Required non-empty string fields
    required_strings = [
        "calculated_tam", "market_analysis_summary",
        "moat_types_present", "moat_evidence", "moat_analysis_summary",
        "problem_severity", "problem_frequency", "demand_evidence",
        "problem_analysis_summary",
        "web_sources_used",
    ]
    for f in required_strings:
        v = result.get(f)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            errors.append(f"{f}: missing or empty")

    # Score coercion + range
    for f in ("market_size_score", "moat_score", "problem_score"):
        result[f] = _coerce_score(result.get(f), f, errors)

    # Confidence enums
    for f in ("market_confidence", "moat_confidence", "problem_confidence"):
        _check_enum(result.get(f), ALLOWED_CONFIDENCE, f, errors)

    # Problem-specific enums
    _check_enum(result.get("problem_severity"), ALLOWED_SEVERITY, "problem_severity", errors)
    _check_enum(result.get("problem_frequency"), ALLOWED_FREQUENCY, "problem_frequency", errors)

    # Anti-laziness: high scores require substantive supporting text
    if isinstance(result.get("moat_score"), int) and result["moat_score"] >= 7:
        _check_substantive(result.get("moat_evidence"), "moat_evidence (for moat_score>=7)", 50, errors)
    if isinstance(result.get("problem_score"), int) and result["problem_score"] >= 7:
        _check_substantive(result.get("demand_evidence"), "demand_evidence (for problem_score>=7)", 30, errors)

    # Halo-effect check: warn (not error) if all three scores collapse to the same value
    scores = [result.get(f) for f in ("market_size_score", "moat_score", "problem_score")]
    if all(isinstance(s, int) for s in scores) and len(set(scores)) == 1:
        errors.append(f"halo-effect suspect: all three scores identical ({scores[0]}) — review prompts/output")

    return errors


def _http_code(err) -> int | None:
    """Best-effort extraction of an HTTP status code from a google-genai APIError."""
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if isinstance(code, int):
        return code
    # Fall back to parsing the leading "NNN" from the str representation
    m = re.match(r"\s*(\d{3})\b", str(err))
    return int(m.group(1)) if m else None


def _gemini_call(contents, max_retries: int = 5) -> dict:
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    base_delay = 10
    last_err = None
    for attempt in range(max_retries):
        try:
            response = get_client().models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            return parse_json_response(response.text)
        except APIError as e:
            last_err = e
            code = _http_code(e)
            # 4xx (except 429 rate-limit) are permanent client errors — fail fast
            # rather than burn 70+ seconds of exponential backoff on a guaranteed loss.
            if code is not None and 400 <= code < 500 and code != 429:
                print(f"    APIError {code} (non-retryable client error): "
                      f"{truncate(str(e), 120)}")
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    APIError on attempt {attempt + 1}"
                  + (f" (HTTP {code})" if code else "")
                  + f": retrying in {delay}s...")
            time.sleep(delay)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"    Parse error on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(5)
    raise RuntimeError(f"All {max_retries} attempts failed: {last_err}")


def evaluate_row(row, deck_bytes=None, deck_mime=None, max_retries: int = 5) -> dict:
    prompt = build_prompt(row, deck_attached=deck_bytes is not None)
    if deck_bytes is not None:
        contents = [
            types.Part.from_bytes(data=deck_bytes, mime_type=deck_mime),
            prompt,
        ]
    else:
        contents = prompt
    return _gemini_call(contents, max_retries=max_retries)


def row_key(row, idx: int) -> str:
    """Stable per-row identifier. Falls back to row index when Tracking ID is missing
    so multiple rows with blank Tracking IDs are not deduped together."""
    tid = safe(row.get("Tracking ID"))
    return tid if tid != "N/A" else f"row_{idx}"


OUTPUT_COLUMNS = [
    # identifiers
    "Tracking ID",
    "Startup Name",
    "Sector",
    "Stage",
    "Pitch Deck URL",
    # market
    "deck_fetch_status",
    "market_size_score",
    "calculated_tam",
    "deck_tam_claim",
    "cagr",
    "geographic_scope",
    "growth_tailwinds",
    "growth_headwinds",
    "market_confidence",
    "market_analysis_summary",
    # moat
    "moat_score",
    "moat_types_present",
    "deck_moat_claim",
    "moat_evidence",
    "moat_risks",
    "moat_confidence",
    "moat_analysis_summary",
    # problem validation
    "problem_score",
    "problem_severity",
    "problem_frequency",
    "existing_willingness_to_pay",
    "demand_evidence",
    "problem_red_flags",
    "problem_confidence",
    "problem_analysis_summary",
    # shared deck signals
    "pitch_deck_accessed",
    "pitch_deck_notes",
    "web_sources_used",
    # diagnostics
    "validation_warnings",
    "evaluation_error",
]


def sanitize_cell(value):
    """Collapse newlines/whitespace in any string going to CSV so rows can't drift."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    # collapse any run of whitespace (incl. \n, \r, \t) into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def load_done_ids() -> set:
    if not RESULTS_PATH.exists():
        return set()
    try:
        existing = pd.read_csv(RESULTS_PATH, dtype={"Tracking ID": str})
        return set(existing["Tracking ID"].dropna().astype(str))
    except Exception as e:
        print(f"Warning: could not read existing results ({e}); starting fresh.")
        return set()


def append_result(out_row: dict) -> None:
    write_header = not RESULTS_PATH.exists()
    clean = {col: sanitize_cell(out_row.get(col)) for col in OUTPUT_COLUMNS}
    pd.DataFrame([clean], columns=OUTPUT_COLUMNS).to_csv(
        RESULTS_PATH,
        mode="a",
        header=write_header,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )


# ---------------------------------------------------------------------------
# Terminal UX helpers
# ---------------------------------------------------------------------------

BANNER = r"""
================================================================================

   ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██╗   ██╗███████╗███████╗████████╗
  ██╔════╝██╔═══██╗████╗  ██║██╔═══██╗██║   ██║██╔════╝██╔════╝╚══██╔══╝
  ██║     ██║   ██║██╔██╗ ██║██║   ██║██║   ██║█████╗  ███████╗   ██║
  ██║     ██║   ██║██║╚██╗██║██║▄▄ ██║██║   ██║██╔══╝  ╚════██║   ██║
  ╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝╚██████╔╝███████╗███████║   ██║
   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝

                       A I   E V A L U A T O R
            Market  ·  MOAT  ·  Problem Validation  ·  Gemini

================================================================================
"""


def fmt_duration(seconds: float) -> str:
    s = int(max(seconds, 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def print_run_header(csv_path: Path, results_path: Path, model: str, limit, start, total_rows: int,
                     done_count: int) -> None:
    print(BANNER)
    print(f"  Input file   : {csv_path.name}  ({total_rows} rows)")
    print(f"  Output file  : {results_path.name}")
    print(f"  Model        : {model}")
    print(f"  Run limit    : {limit if limit is not None else 'none (all rows)'}")
    if start:
        print(f"  Start offset : skip first {start} rows")
    if done_count:
        print(f"  Resuming     : {done_count} rows already scored, will skip those")
    print("=" * 80)
    print()


def print_row_header(local_pos: int, local_total, global_idx: int, global_total: int,
                     elapsed: float, name: str, sector: str, stage: str) -> None:
    pct = 100.0 * local_pos / local_total if local_total else 0.0
    progress_bar = _bar(local_pos, local_total, width=24) if local_total else ""
    print()
    print("─" * 80)
    if local_total:
        print(f"  [{local_pos}/{local_total}]  {progress_bar}  {pct:5.1f}%   "
              f"row {global_idx} of {global_total}   elapsed {fmt_duration(elapsed)}")
    else:
        print(f"  [{local_pos}]  row {global_idx} of {global_total}   "
              f"elapsed {fmt_duration(elapsed)}")
    print("─" * 80)
    print(f"  Startup : {truncate(name, 70)}")
    print(f"  Sector  : {truncate(sector, 40)}   |   Stage: {truncate(stage, 20)}")


def _bar(done: int, total: int, width: int = 24) -> str:
    if not total:
        return ""
    filled = int(width * done / total)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def print_eta(local_pos: int, local_total, elapsed: float) -> None:
    if not local_total or local_pos < 1:
        return
    avg = elapsed / local_pos
    remaining = avg * (local_total - local_pos)
    print(f"  ETA     : ~{fmt_duration(remaining)} remaining   "
          f"({avg:.1f}s/row average)")


def print_run_summary(*, processed: int, skipped_empty: int, errors: int,
                      rows_with_warnings: int, fetchable: int, downloaded: int,
                      deck_accessed: int, elapsed: float, results_path: Path) -> None:
    print()
    print("=" * 80)
    print("  R U N   C O M P L E T E")
    print("=" * 80)
    print(f"  Startups processed        : {processed}")
    print(f"  Empty rows skipped        : {skipped_empty}")
    print(f"  Validation warnings (rows): {rows_with_warnings}")
    print(f"  Evaluation errors         : {errors}")
    print(f"  Decks fetchable           : {fetchable}")
    print(f"  Decks downloaded          : {downloaded}"
          + (f"  ({100.0 * downloaded / fetchable:.0f}% success)" if fetchable else ""))
    print(f"  Decks read by Gemini      : {deck_accessed}")
    print(f"  Total elapsed             : {fmt_duration(elapsed)}")
    print(f"  Output saved to           : {results_path.name}")
    print("=" * 80)


def main():
    global RESULTS_PATH

    parser = argparse.ArgumentParser(
        description="Score startups on Market / MOAT / Problem Validation / Founder using Gemini."
    )
    parser.add_argument("--csv", type=str, default=str(CSV_PATH),
                        help=f"Path to input CSV (default: {CSV_PATH.name}).")
    parser.add_argument("--output", type=str, default=str(RESULTS_PATH),
                        help=f"Path to output CSV (default: {RESULTS_PATH.name}). "
                             "Appended to incrementally; resumable on re-run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many NEW rows in this run.")
    parser.add_argument("--only-with-deck", action="store_true",
                        help="Skip rows that have no Pitch Deck URL.")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip the first N rows of the source CSV before processing.")
    args = parser.parse_args()

    # Apply CLI overrides for input/output paths
    csv_path = Path(args.csv).expanduser().resolve()
    RESULTS_PATH = Path(args.output).expanduser().resolve()

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment before running.")

    df = pd.read_csv(csv_path, dtype={"Tracking ID": str})

    done_ids = load_done_ids()

    work = df.iloc[args.start:]
    if args.only_with_deck:
        work = work[work["Pitch Deck URL"].notna()]

    print_run_header(csv_path, RESULTS_PATH, MODEL,
                     limit=args.limit, start=args.start,
                     total_rows=len(df), done_count=len(done_ids))
    if args.only_with_deck:
        print(f"  --only-with-deck active: {len(work)} candidate rows have a pitch deck URL.")
        print()

    # Pre-compute how many rows will actually be processed this run (for progress %)
    candidates = []
    for idx, row in work.iterrows():
        name = safe(row.get("Startup Name"))
        if name == "N/A":
            continue
        tid = row_key(row, idx)
        if tid in done_ids:
            continue
        candidates.append((idx, row, tid, name))
    local_total = min(len(candidates), args.limit) if args.limit is not None else len(candidates)

    if local_total == 0:
        print("  Nothing to do — all eligible rows are already in the output file.")
        return

    processed = 0
    skipped_empty = sum(1 for _, r in work.iterrows() if safe(r.get("Startup Name")) == "N/A")
    fetchable = 0
    downloaded = 0
    deck_accessed = 0
    errors = 0
    rows_with_warnings = 0
    run_start = time.time()

    for (idx, row, tid, name) in candidates:
        if processed >= local_total:
            break

        elapsed = time.time() - run_start
        deck_url = safe(row.get("Pitch Deck URL"))
        sector = safe(row.get("Sector"))
        stage = safe(row.get("Stage"))

        print_row_header(processed + 1, local_total, idx + 1, len(df),
                         elapsed, name, sector, stage)

        out = {col: None for col in OUTPUT_COLUMNS}
        out.update({
            "Tracking ID": tid,
            "Startup Name": name,
            "Sector": sector,
            "Stage": stage,
            "Pitch Deck URL": deck_url,
        })

        # ---------- Pitch deck download ----------
        deck_bytes = deck_mime = None
        if is_fetchable_deck(deck_url):
            fetchable += 1
            deck_bytes, deck_mime, status = download_deck(deck_url)
            out["deck_fetch_status"] = status
            if deck_bytes is not None:
                downloaded += 1
                size_kb = len(deck_bytes) / 1024
                print(f"  → Deck    : downloaded {size_kb:,.0f} KB")
            else:
                print(f"  → Deck    : {status}")
        elif deck_url == "N/A":
            out["deck_fetch_status"] = "no url"
            print(f"  → Deck    : no URL provided")
        else:
            out["deck_fetch_status"] = "skipped (unsupported host)"
            print(f"  → Deck    : skipped (unsupported host)")

        # ---------- Market + MOAT + Problem (single grounded call) ----------
        print(f"  → Gemini  : calling (market + MOAT + problem)...")
        try:
            result = evaluate_row(row, deck_bytes=deck_bytes, deck_mime=deck_mime)
            warnings = validate_market_moat_problem(result)
            if warnings:
                rows_with_warnings += 1
                out["validation_warnings"] = " | ".join(warnings)

            out.update({
                "market_size_score": result.get("market_size_score"),
                "calculated_tam": result.get("calculated_tam"),
                "deck_tam_claim": result.get("deck_tam_claim"),
                "cagr": result.get("cagr"),
                "geographic_scope": result.get("geographic_scope"),
                "growth_tailwinds": result.get("growth_tailwinds"),
                "growth_headwinds": result.get("growth_headwinds"),
                "market_confidence": result.get("market_confidence"),
                "market_analysis_summary": result.get("market_analysis_summary"),
                "moat_score": result.get("moat_score"),
                "moat_types_present": result.get("moat_types_present"),
                "deck_moat_claim": result.get("deck_moat_claim"),
                "moat_evidence": result.get("moat_evidence"),
                "moat_risks": result.get("moat_risks"),
                "moat_confidence": result.get("moat_confidence"),
                "moat_analysis_summary": result.get("moat_analysis_summary"),
                "problem_score": result.get("problem_score"),
                "problem_severity": result.get("problem_severity"),
                "problem_frequency": result.get("problem_frequency"),
                "existing_willingness_to_pay": result.get("existing_willingness_to_pay"),
                "demand_evidence": result.get("demand_evidence"),
                "problem_red_flags": result.get("problem_red_flags"),
                "problem_confidence": result.get("problem_confidence"),
                "problem_analysis_summary": result.get("problem_analysis_summary"),
                "pitch_deck_accessed": result.get("pitch_deck_accessed"),
                "pitch_deck_notes": result.get("pitch_deck_notes"),
                "web_sources_used": result.get("web_sources_used"),
            })
            if bool(out["pitch_deck_accessed"]):
                deck_accessed += 1

            tam_brief = truncate(str(out["calculated_tam"]), 60)
            moats_brief = truncate(str(out["moat_types_present"]), 50)
            problem_brief = f"{out['problem_severity']} / {out['problem_frequency']}"
            print(f"  ✓ Market  : {out['market_size_score']}/10  "
                  f"({out['market_confidence']:<6})   TAM: {tam_brief}")
            print(f"  ✓ MOAT    : {out['moat_score']}/10  "
                  f"({out['moat_confidence']:<6})   {moats_brief}")
            print(f"  ✓ Problem : {out['problem_score']}/10  "
                  f"({out['problem_confidence']:<6})   {problem_brief}")

            if warnings:
                print(f"  ⚠ Warnings: {len(warnings)} — {truncate(warnings[0], 60)}"
                      + (f"  (+{len(warnings)-1} more)" if len(warnings) > 1 else ""))
        except Exception as e:
            errors += 1
            out["evaluation_error"] = str(e)
            print(f"  ✗ ERROR   : {truncate(str(e), 70)}")

        append_result(out)
        done_ids.add(tid)
        processed += 1
        print(f"  ✓ Saved   : appended to {RESULTS_PATH.name}")
        print_eta(processed, local_total, time.time() - run_start)
        time.sleep(API_PAUSE_SECONDS)

    print_run_summary(
        processed=processed,
        skipped_empty=skipped_empty,
        errors=errors,
        rows_with_warnings=rows_with_warnings,
        fetchable=fetchable,
        downloaded=downloaded,
        deck_accessed=deck_accessed,
        elapsed=time.time() - run_start,
        results_path=RESULTS_PATH,
    )


if __name__ == "__main__":
    main()
