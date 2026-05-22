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
CSV_PATH = SCRIPT_DIR / "conquest-registrations-2026-04-25 (2).csv"
RESULTS_PATH = SCRIPT_DIR / "evaluate_market_results.csv"

MODEL = "gemini-2.5-flash"
API_PAUSE_SECONDS = 1.0

# Only attempt to download decks from these hosts. Others (OneDrive, Google
# Drive share links, etc.) are auth-gated and can't be fetched reliably.
FETCHABLE_DECK_HOSTS = ("conquestbits.org",)
MAX_DECK_BYTES = 20 * 1024 * 1024  # 20 MB cap before we skip an attachment
DECK_DOWNLOAD_TIMEOUT = 30

# LinkedIn fetching. LinkedIn aggressively blocks bots so success rate will be
# low; the prompt is built to fall back to Google Search when content is thin.
LINKEDIN_FETCH_TIMEOUT = 20
LINKEDIN_MAX_TEXT_CHARS = 15000  # cap to keep prompt size reasonable
LINKEDIN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

client = genai.Client()


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
You are an elite venture capital analyst performing automated due diligence.
Produce TWO independent integer scores (1-10) for the startup below:
  A) MARKET SIZE
  B) DIFFERENTIATION / MOAT

STARTUP PROFILE
- Company Name: {safe(row.get('Startup Name'))}
- Sector: {safe(row.get('Sector'))}
- Business Model: {safe(row.get('Business Model'))}
- Current Stage: {safe(row.get('Stage'))}
- Location: {safe(row.get('Location'))}
- Startup Age: {safe(row.get('Startup Age'))}
- Team Size: {safe(row.get('Team Size'))}
- Founder LinkedIn: {safe(row.get('LinkedIn'))}
- Pitch Deck URL: {safe(row.get('Pitch Deck URL'))}

FOUNDER DESCRIPTION
- Problem Being Solved: {safe(row.get('Problem Solving'))}
- Solution Provided: {safe(row.get('Solution'))}
- Target Customer Segment: {safe(row.get('Target Customer'))}
- USPs: {safe(row.get('USPs'))}
- Achievements: {safe(row.get('Achievements'))}
- Has Patents: {safe(row.get('Has Patents'))}
- Patent Details: {safe(row.get('Patent Details'))}

PITCH DECK HANDLING
{deck_instruction}

MANDATORY RESEARCH METHOD (do ALL of these for EVERY startup, deck or no deck):
1. Use the Google Search tool. At minimum search for:
   a) the startup name itself (news, funding, coverage)
   b) the sector + "market size" / "TAM" / "CAGR" with the latest year
   c) 1-3 comparable companies in the same space (for both sizing-by-analogy AND competitive density)
   d) regional reports if India-focused (Inc42, RedSeer, Tracxn, Bain India)
   e) for moat: search the company name + "patent" / "IP" / "open source" / "API" to verify defensibility claims
2. Form your OWN estimates. Do NOT just copy numbers/claims from the deck or the first search hit.
3. If the deck disagrees with your independent estimate, USE YOURS and explain the gap.

==== A) MARKET SIZE SCORING — multi-factor, NOT just raw dollars ====
Weigh ALL of these, then assign one integer 1-10:
  (a) Size of the realistically addressable market for THIS startup's wedge (SAM, not the whole sector's TAM)
  (b) Growth rate / CAGR — a $2B market growing 30% YoY can beat a $50B stagnant one
  (c) Tailwinds (regulation, tech shift, capital inflow, demographics, behavior change)
  (d) Headwinds (entrenched incumbents, commoditization, regulatory risk, capital-intensity)
  (e) Geographic reach (India-only vs globally expandable)
  (f) Timing — is the market forming, growing, mature, or saturating?

Market rubric (apply AFTER weighing all factors above):
  1-2  = tiny niche or shrinking; no realistic upside
  3-4  = small / fragmented, weak growth or hostile dynamics
  5-6  = solid mid-size market (~$1-5B SAM), moderate growth, ordinary dynamics
  7-8  = large market ($5-20B SAM) with strong growth or major tailwinds
  9-10 = massive (>$20B) hyper-growth or category-defining inflection

==== B) DIFFERENTIATION / MOAT SCORING ====
Assess how defensible this startup actually is against well-funded competition.
Identify which (if any) of these moat TYPES are genuinely present — be skeptical:
  - Tech / IP moat: granted patents (verify), proprietary algorithms, hard-to-replicate engineering
  - Data moat: proprietary dataset compounding with use; network effect on data
  - Network effects: more users -> more value for other users (genuine two-sided dynamic, not just "marketplace")
  - Brand / community: defensible mindshare or a strong organic community
  - Switching costs / lock-in: integrations, workflows, data trapped in product
  - Regulatory moat: licenses, certifications, regulated activity others can't enter easily
  - Distribution moat: exclusive channels, deep partnerships, embedded into customer ops
  - Scale / cost: structural cost advantage from scale

For each claimed moat, ask: is there EVIDENCE, or is it just a USP slide? Filed patents != granted; "AI-powered" != tech moat; "marketplace" != network effects.

Moat rubric:
  1-2  = commodity offering in a crowded space; no real differentiation; trivially replicable
  3-4  = some differentiation but easily copied; weak or temporary moats; mostly feature-level diff
  5-6  = ONE real but vulnerable moat (e.g. brand, modest data, partnerships) that a well-funded entrant could erode
  7-8  = TWO+ meaningful moats compounding (e.g. tech IP + distribution; data + switching costs)
  9-10 = step-change defensibility incumbents cannot easily replicate (genuine network effects at scale, granted patents in active use, structural data lead, regulatory monopoly)

OUTPUT
Return ONLY a raw JSON object (no markdown fences, no prose outside JSON), in EXACTLY this schema.
All string values MUST be single-line (no newlines inside strings):
{{
  "market_size_score": 7,
  "calculated_tam": "Your INDEPENDENT TAM, e.g. '$12B SAM by 2030 (India)' + 1-line basis",
  "deck_tam_claim": "What the deck/founder claimed, or 'not stated' / 'not available'",
  "cagr": "Your researched CAGR + year range, e.g. '18% (2024-2030)', or 'unknown'",
  "growth_tailwinds": "1-2 key tailwinds, single line",
  "growth_headwinds": "1-2 key headwinds, single line",
  "geographic_scope": "e.g. 'India-only', 'India + SEA', 'Global'",
  "market_analysis_summary": "3-5 sentences integrating deck (if any) + web research, explaining why this score and not one notch higher or lower.",
  "market_confidence": "low | medium | high",

  "moat_score": 6,
  "moat_types_present": "Comma list of moats that are GENUINELY present with evidence, e.g. 'Tech IP, Distribution'. Empty string if none.",
  "deck_moat_claim": "What the deck/founder claimed as their moat or USP, or 'not stated' / 'not available'",
  "moat_evidence": "The specific evidence supporting the moats above (e.g. 'Patent US-12345 granted 2023', 'Exclusive partnership with HDFC verified via press release')",
  "moat_risks": "What would erode the moats — 1-2 sentences",
  "moat_analysis_summary": "3-5 sentences justifying this exact score. Single line.",
  "moat_confidence": "low | medium | high",

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

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        mime = "application/pdf"
    elif content_type:
        mime = content_type
    else:
        mime = "application/pdf"  # best guess for the conquestbits endpoint

    return data, mime, f"downloaded {len(data)} bytes ({mime})"


def normalize_linkedin_url(url: str) -> str | None:
    if url == "N/A" or not url:
        return None
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url.lstrip("/")
    return url


def _strip_html(html: str) -> str:
    """Cheap HTML -> text. No bs4 dependency."""
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<noscript.*?</noscript>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    # decode the most common HTML entities
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
            .replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#39;", "'"))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_linkedin_text(url: str):
    """Return (text or None, status_msg).
    If LinkedIn blocks, returns no usable HTML, or only serves a login wall, we
    return None and founder scoring is skipped entirely (founder fields stay NA).
    We never fall back to Google Search for founder data.
    """
    try:
        resp = requests.get(
            url, headers=LINKEDIN_HEADERS,
            timeout=LINKEDIN_FETCH_TIMEOUT, allow_redirects=True,
        )
    except requests.RequestException as e:
        return None, f"fetch failed: {e.__class__.__name__}"

    if resp.status_code != 200:
        return None, f"fetch failed: HTTP {resp.status_code}"

    text = _strip_html(resp.text)
    # Heuristic: detect login wall / interstitial -- treat as failure (no signal).
    login_markers = (
        "sign in to linkedin", "join linkedin", "join now to see",
        "to view this profile", "authwall",
    )
    lowered = text[:2000].lower()
    if any(marker in lowered for marker in login_markers) and len(text) < 4000:
        return None, "login wall (no usable profile content)"

    if len(text) < 200:
        return None, f"empty/thin response ({len(text)} chars)"

    return text[:LINKEDIN_MAX_TEXT_CHARS], f"fetched {len(text)} chars"


def build_founder_prompt(row, linkedin_text: str) -> str:
    """Build the founder-eval prompt. Only called when we have real LinkedIn content."""
    founder_name = safe(row.get("Full Name"))
    startup_name = safe(row.get("Startup Name"))
    sector = safe(row.get("Sector"))
    linkedin_url = safe(row.get("LinkedIn"))

    return f"""
You are an elite venture capital analyst evaluating a startup's PRIMARY FOUNDER
based STRICTLY on the LinkedIn profile content provided below. Do NOT use any
other source, do NOT search the web, do NOT speculate beyond what is in the text.

FOUNDER
- Name: {founder_name}
- LinkedIn URL: {linkedin_url}
- Startup: {startup_name} (Sector: {sector})

LINKEDIN PAGE CONTENT (scraped HTML, cleaned to text):
---BEGIN---
{linkedin_text}
---END---

The content may include navigation chrome and noise — extract the founder-specific signal only.
If a fact (school, year, role) is not stated in the content above, mark it 'unknown'. Do not infer.

FOCUS AREAS
- Pedigree: schools attended + degrees
- Prior companies: company name + role + tenure
- PRIOR FOUNDING EXPERIENCE (especially exits)
- Domain expertise relevant to the startup's sector ({sector})
- Total years of relevant work experience
- Leadership scope (team size, scale of responsibility)

SCORING RUBRIC (integer 1-10)
  9-10 = Repeat founder with prior EXIT, OR top-tier pedigree (IIT/IIM/Stanford/MIT/Wharton/Harvard) + senior role (5+ yrs) at a top tech company + deep domain expertise in this startup's sector
  7-8  = Strong pedigree (top school OR top company) + prior startup experience OR 5+ yrs leadership + clear domain fit
  5-6  = Decent background, mid-tier school OR company, 3-5 yrs relevant work, some domain fit
  3-4  = Thin credentials, recent graduate, limited relevant experience, weak or no domain fit
  1-2  = No verifiable background, irrelevant experience, or red flags

If the LinkedIn content is sparse, score conservatively. Do NOT default to a mid score.

OUTPUT
Return ONLY a raw JSON object, no markdown, all strings single-line:
{{
  "founder_score": 7,
  "founder_education": "School(s) + degree, e.g. 'IIT Bombay B.Tech CS 2015', or 'unknown'",
  "founder_companies": "Comma list of prior companies + role + years, or 'unknown'",
  "prior_founding_experience": "Yes/No + brief detail, or 'No'",
  "domain_fit": "How well does background match the startup's sector? 1-2 sentences",
  "years_relevant_experience": "Integer or short range, or 'unknown'",
  "founder_red_flags": "Any concerns, or 'none observed'",
  "founder_analysis_summary": "3-5 sentences justifying the score, citing only LinkedIn content. Single line.",
  "founder_confidence": "low | medium | high"
}}
""".strip()


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


def _gemini_call(contents, use_search: bool = True, max_retries: int = 5) -> dict:
    if use_search:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    else:
        config = types.GenerateContentConfig()

    base_delay = 10
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            return parse_json_response(response.text)
        except APIError as e:
            last_err = e
            delay = base_delay * (2 ** attempt)
            print(f"    APIError on attempt {attempt + 1}: {e}. Retrying in {delay}s...")
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
    # Market + MOAT uses Google Search for live sector research.
    return _gemini_call(contents, use_search=True, max_retries=max_retries)


def evaluate_founder(row, linkedin_text: str, max_retries: int = 5) -> dict:
    prompt = build_founder_prompt(row, linkedin_text)
    # Founder scoring is STRICTLY LinkedIn-only -- no Google Search.
    return _gemini_call(prompt, use_search=False, max_retries=max_retries)


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
    "LinkedIn",
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
    # shared deck signals
    "pitch_deck_accessed",
    "pitch_deck_notes",
    "web_sources_used",
    # founder
    "linkedin_fetch_status",
    "founder_score",
    "founder_education",
    "founder_companies",
    "prior_founding_experience",
    "domain_fit",
    "years_relevant_experience",
    "founder_red_flags",
    "founder_confidence",
    "founder_analysis_summary",
    # diagnostics
    "market_moat_error",
    "founder_error",
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


def main():
    parser = argparse.ArgumentParser(description="Score market size for each startup using Gemini.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many NEW rows in this run.")
    parser.add_argument("--only-with-deck", action="store_true",
                        help="Skip rows that have no Pitch Deck URL.")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip the first N rows of the source CSV before processing.")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}")
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment before running.")

    df = pd.read_csv(CSV_PATH, dtype={"Tracking ID": str})
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    done_ids = load_done_ids()
    if done_ids:
        print(f"Resuming: {len(done_ids)} rows already scored; will skip those.")

    work = df.iloc[args.start:]
    if args.only_with_deck:
        work = work[work["Pitch Deck URL"].notna()]
        print(f"--only-with-deck active: {len(work)} candidate rows have a pitch deck URL.")

    processed = 0
    skipped_empty = 0
    fetchable = 0
    downloaded = 0
    deck_accessed = 0
    market_errors = 0
    linkedin_present = 0
    linkedin_fetched = 0
    founder_scored = 0
    founder_errors = 0

    for idx, row in work.iterrows():
        name = safe(row.get("Startup Name"))
        # Skip rows with no Startup Name (e.g., abandoned/empty registrations).
        if name == "N/A":
            skipped_empty += 1
            continue

        # Stable per-row key. Synthetic when Tracking ID is missing so rows with
        # blank tids don't all collapse to the same dedup bucket.
        tid = row_key(row, idx)
        if tid in done_ids:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        deck_url = safe(row.get("Pitch Deck URL"))
        linkedin_url_raw = safe(row.get("LinkedIn"))
        print(f"\n[{idx + 1}/{len(df)}] {name}")

        out = {col: None for col in OUTPUT_COLUMNS}
        out.update({
            "Tracking ID": tid,
            "Startup Name": name,
            "Sector": safe(row.get("Sector")),
            "Stage": safe(row.get("Stage")),
            "Pitch Deck URL": deck_url,
            "LinkedIn": linkedin_url_raw,
        })

        # ---------- Pitch deck download ----------
        deck_bytes = deck_mime = None
        if is_fetchable_deck(deck_url):
            fetchable += 1
            print(f"    fetching deck: {deck_url}")
            deck_bytes, deck_mime, status = download_deck(deck_url)
            out["deck_fetch_status"] = status
            print(f"    {status}")
            if deck_bytes is not None:
                downloaded += 1
        elif deck_url == "N/A":
            out["deck_fetch_status"] = "no url"
        else:
            out["deck_fetch_status"] = "skipped (unsupported host)"

        # ---------- Market + MOAT (single combined call) ----------
        try:
            result = evaluate_row(row, deck_bytes=deck_bytes, deck_mime=deck_mime)
            out.update({
                # market
                "market_size_score": result.get("market_size_score"),
                "calculated_tam": result.get("calculated_tam"),
                "deck_tam_claim": result.get("deck_tam_claim"),
                "cagr": result.get("cagr"),
                "geographic_scope": result.get("geographic_scope"),
                "growth_tailwinds": result.get("growth_tailwinds"),
                "growth_headwinds": result.get("growth_headwinds"),
                "market_confidence": result.get("market_confidence"),
                "market_analysis_summary": result.get("market_analysis_summary"),
                # moat
                "moat_score": result.get("moat_score"),
                "moat_types_present": result.get("moat_types_present"),
                "deck_moat_claim": result.get("deck_moat_claim"),
                "moat_evidence": result.get("moat_evidence"),
                "moat_risks": result.get("moat_risks"),
                "moat_confidence": result.get("moat_confidence"),
                "moat_analysis_summary": result.get("moat_analysis_summary"),
                # shared
                "pitch_deck_accessed": result.get("pitch_deck_accessed"),
                "pitch_deck_notes": result.get("pitch_deck_notes"),
                "web_sources_used": result.get("web_sources_used"),
            })
            print(f"    market={out['market_size_score']}  moat={out['moat_score']}  "
                  f"deck_tam={out['deck_tam_claim']}  indep_tam={out['calculated_tam']}  "
                  f"deck_read={out['pitch_deck_accessed']}")
            if bool(out["pitch_deck_accessed"]):
                deck_accessed += 1
        except Exception as e:
            market_errors += 1
            out["market_moat_error"] = str(e)
            print(f"    MARKET/MOAT ERROR: {e}")

        # ---------- Founder (LinkedIn-only; NA when LinkedIn unavailable) ----------
        # Strict rule: founder score comes from LinkedIn content only. If we can't
        # get LinkedIn (no URL, blocked, login wall, etc.), founder fields stay NA.
        # No Google Search fallback -- it's unreliable for founder data.
        linkedin_url = normalize_linkedin_url(linkedin_url_raw)
        if linkedin_url is None:
            out["linkedin_fetch_status"] = "no linkedin url -> founder NA"
            print(f"    no LinkedIn URL -> founder NA")
        else:
            linkedin_present += 1
            print(f"    fetching LinkedIn: {linkedin_url}")
            linkedin_text, linkedin_status = fetch_linkedin_text(linkedin_url)
            out["linkedin_fetch_status"] = linkedin_status
            print(f"    linkedin: {linkedin_status}")
            if linkedin_text is None:
                print(f"    LinkedIn unreadable -> founder NA")
            else:
                linkedin_fetched += 1
                try:
                    f_result = evaluate_founder(row, linkedin_text)
                    out.update({
                        "founder_score": f_result.get("founder_score"),
                        "founder_education": f_result.get("founder_education"),
                        "founder_companies": f_result.get("founder_companies"),
                        "prior_founding_experience": f_result.get("prior_founding_experience"),
                        "domain_fit": f_result.get("domain_fit"),
                        "years_relevant_experience": f_result.get("years_relevant_experience"),
                        "founder_red_flags": f_result.get("founder_red_flags"),
                        "founder_confidence": f_result.get("founder_confidence"),
                        "founder_analysis_summary": f_result.get("founder_analysis_summary"),
                    })
                    founder_scored += 1
                    print(f"    founder={out['founder_score']}  edu={out['founder_education']}  "
                          f"prior_founding={out['prior_founding_experience']}  conf={out['founder_confidence']}")
                except Exception as e:
                    founder_errors += 1
                    out["founder_error"] = str(e)
                    print(f"    FOUNDER ERROR: {e}")

        append_result(out)
        done_ids.add(tid)
        processed += 1
        time.sleep(API_PAUSE_SECONDS)

    print("\n=== Run summary ===")
    print(f"Processed this run:                {processed}")
    print(f"Empty rows skipped (no name):      {skipped_empty}")
    print(f"Market/MOAT errors:                {market_errors}")
    print(f"Founder errors:                    {founder_errors}")
    print(f"Decks fetchable (conquest):        {fetchable}")
    print(f"Decks successfully downloaded:     {downloaded}")
    print(f"Decks model confirmed read:        {deck_accessed}")
    print(f"Rows with LinkedIn URL:            {linkedin_present}")
    print(f"LinkedIn pages with usable text:   {linkedin_fetched}")
    print(f"Founders scored:                   {founder_scored}")
    if fetchable:
        print(f"Deck download success rate:        {100.0 * downloaded / fetchable:.1f}%")
    if linkedin_present:
        print(f"LinkedIn fetch success rate:       {100.0 * linkedin_fetched / linkedin_present:.1f}%")
    print(f"Results file:                      {RESULTS_PATH}")


if __name__ == "__main__":
    main()
