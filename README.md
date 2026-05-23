# AI Startup Evaluator

A Python script that uses the **Gemini API** to evaluate startups along four dimensions:

| Score | What it measures | Inputs used |
|---|---|---|
| **Market Size** (1–10) | Realistic SAM, growth/CAGR, tailwinds, headwinds, geographic reach, timing | Founder text + pitch deck PDF (if available) + live Google Search |
| **Differentiation / MOAT** (1–10) | Tech/IP, data, network effects, brand, switching costs, regulatory, distribution, scale advantages | Same as above; cross-checks deck claims against independent research |
| **Problem Validation** (1–10) | Severity, frequency, prevalence, existing willingness-to-pay, articulation quality, demand evidence | Same as above |
| **Founder Profile** (1–10) | Pedigree, prior companies, founding experience, domain fit, years of relevant experience | **LinkedIn profile content only** (no web search). Returns NA if LinkedIn cannot be fetched. |

Market + MOAT + Problem Validation are produced by **one** Gemini call per startup (with Google Search grounding). Founder is a **separate** call (no grounding) when LinkedIn content is available. Results are appended to a CSV one row at a time, so the script is **crash-safe and resumable**.

### Anti-bias & quality controls

The prompts include explicit guardrails: sector-neutral, geography-neutral, founder-identity-neutral for the business scores, no halo effect across dimensions, buzzwords-without-evidence count as zero, evidence-only for founder scoring, and conservative defaults when data is thin. Every Gemini response is run through a **validator** that checks score type/range, required fields, confidence-enum values, evidence-substance for high scores, and a halo-effect heuristic. Any violations are written to `validation_warnings_*` columns in the CSV for auditing — they never block writing the row.

---

## 1. Prerequisites

- **Python 3.10+** (`python3 --version` to check)
- **A Gemini API key** (free tier works for testing; paid tier needed for the full dataset — see step 3)
- A **registration CSV** in this directory with the expected columns (see [Input CSV format](#input-csv-format) below)

---

## 2. Setup

### 2.1 Clone the repo and install dependencies

```bash
git clone https://github.com/priyansh-x/Ai-eval.git
cd Ai-eval
pip install -r requirements.txt
```

### 2.2 Get a Gemini API key

1. Go to **https://aistudio.google.com/app/apikey**
2. Sign in with a Google account
3. Click **"Create API key"** → select or create a Google Cloud project
4. Copy the key (it will look like `AIzaSy...`)

> **Free tier limits (Gemini 2.5 Flash):** roughly 20 grounded requests/day. Each startup costs ~1 grounded API call (plus 1 ungrounded call if its LinkedIn URL is fetchable). For more than ~15–20 startups/day you need to **enable billing** in your Google Cloud project: https://aistudio.google.com/app/billing — Gemini 2.5 Flash is very cheap (the full 669-row dataset costs well under $10).

### 2.3 Export the API key into your shell

```bash
export GEMINI_API_KEY="AIzaSy...your_key_here..."
```

To make this persistent across terminal sessions, add the same line to `~/.zshrc` (macOS default) or `~/.bashrc` (Linux).

Verify it's set:

```bash
echo $GEMINI_API_KEY
```

### 2.4 Place your input CSV

Drop your registration CSV into the project root. The script looks for the filename hardcoded at the top of `evaluate_market.py`:

```python
CSV_PATH = SCRIPT_DIR / "conquest-registrations-2026-04-25 (2).csv"
```

If your file is named differently, update that line.

> **Privacy note:** the CSV is `.gitignore`d on purpose — it contains founder names, emails, and phone numbers. Never commit it to a public repo.

---

## 3. Running the script

### 3.1 Safe first run — 15 startups only

For your first run (and to stay well within the free tier's daily quota), evaluate just **15 startups**:

```bash
python3 evaluate_market.py --limit 15
```

This will:
- Read the input CSV
- For each of the first 15 startups (skipping rows already evaluated in a prior run, and skipping empty rows):
  1. Download the pitch deck PDF if it's hosted on `conquestbits.org` (in-memory; not saved to disk)
  2. Call Gemini once with the deck + founder text + Google Search grounding → returns **Market** + **MOAT** scores
  3. If a LinkedIn URL is present, fetch it via `requests` → if usable content comes back, call Gemini again (no web search) → returns **Founder** score. Otherwise founder fields stay NA.
- Append each result as a row in `evaluate_market_results.csv`
- Print a per-startup log line and a final run summary

### 3.2 Other useful flags

```bash
# Skip the first N rows in the source CSV (e.g. skip the first 100, then take 15)
python3 evaluate_market.py --start 100 --limit 15

# Only evaluate rows that have a Pitch Deck URL (saves API calls)
python3 evaluate_market.py --only-with-deck --limit 15

# Full run (only do this on the paid tier — will hit free-tier quota almost immediately)
python3 evaluate_market.py
```

### 3.3 Resuming

The script remembers which rows it has already scored (by Tracking ID, or by row index when the Tracking ID is blank). If you stop it with `Ctrl+C` and re-run, it picks up where it left off — **safe to interrupt at any time**.

To start fresh, delete the results file:

```bash
rm evaluate_market_results.csv
```

---

## 4. Output

The script writes `evaluate_market_results.csv` with these columns:

**Identifiers**
- `Tracking ID`, `Startup Name`, `Sector`, `Stage`, `Pitch Deck URL`, `LinkedIn`

**Market**
- `market_size_score` (1–10)
- `calculated_tam` — Gemini's **independent** TAM estimate
- `deck_tam_claim` — what the deck/founder claimed (so you can see the gap)
- `cagr`, `geographic_scope`, `growth_tailwinds`, `growth_headwinds`
- `market_confidence` (low/medium/high)
- `market_analysis_summary`

**MOAT**
- `moat_score` (1–10)
- `moat_types_present` — e.g. *"Tech IP, Distribution"*
- `deck_moat_claim` — founder's claimed USP
- `moat_evidence` — concrete evidence supporting the moats (required substantive when score ≥ 7)
- `moat_risks` — what would erode the moats
- `moat_confidence`, `moat_analysis_summary`

**Problem Validation**
- `problem_score` (1–10)
- `problem_severity` — `high` / `medium` / `low`
- `problem_frequency` — `daily` / `weekly` / `monthly` / `yearly` / `one-time` / `unclear`
- `existing_willingness_to_pay` — is there already spend on workarounds?
- `demand_evidence` — concrete traction / signals (required substantive when score ≥ 7)
- `problem_red_flags` — e.g. *"solution looking for a problem"*
- `problem_confidence`, `problem_analysis_summary`

**Founder**
- `linkedin_fetch_status` — `fetched`, `login wall`, `HTTP 999`, `no linkedin url -> founder NA`, etc.
- `founder_score` (1–10, or blank if LinkedIn unreadable)
- `founder_education`, `founder_companies`, `prior_founding_experience`, `domain_fit`, `years_relevant_experience`, `founder_red_flags`
- `founder_confidence`, `founder_analysis_summary`

**Diagnostics**
- `deck_fetch_status` — `downloaded N bytes`, `skipped (unsupported host)`, `no url`, etc.
- `pitch_deck_accessed`, `pitch_deck_notes`, `web_sources_used`
- `validation_warnings_market_moat_problem` — pipe-separated list of any schema/quality issues with the combined call's output
- `validation_warnings_founder` — same, for the founder call
- `market_moat_error`, `founder_error` — populated only if a call failed after all retries

---

## 5. How each score is computed (high level)

### Market Size
Multi-factor — **not** just raw dollar size. Weighs all of:
- Realistic SAM for this startup's wedge (not the whole sector's TAM)
- CAGR — a $2B market growing 30% YoY can beat a $50B stagnant one
- Tailwinds (regulation, tech shifts, demographics, capital inflow)
- Headwinds (incumbents, commoditization, regulatory risk)
- Geographic reach (India-only vs globally expandable)
- Timing (forming / growing / mature / saturating)

The prompt explicitly tells Gemini that founders systematically overstate TAM, so it produces its own independent estimate and only uses the deck's number as one data point to cross-check.

### MOAT
Skeptical assessment of 8 moat categories:
- Tech / IP (granted patents, hard-to-replicate engineering)
- Data (proprietary dataset compounding with use)
- Network effects (genuine two-sided dynamic)
- Brand / community
- Switching costs / lock-in
- Regulatory (licenses, certifications)
- Distribution (exclusive channels, partnerships)
- Scale / cost advantage

The prompt instructs: *"For each claimed moat, ask: is there EVIDENCE, or is it just a USP slide? Filed patents != granted; 'AI-powered' != tech moat; 'marketplace' != network effects."*

### Problem Validation
Assesses how REAL, URGENT, and VALIDATED the problem is — orthogonal to market size (a huge market can have weakly-validated problems, and vice versa). Evaluates:
- Problem severity (painkiller vs vitamin)
- Frequency (daily / weekly / monthly / yearly)
- Prevalence (how many users)
- Existing willingness-to-pay (current spend on workarounds is the strongest signal)
- Articulation quality (does the founder describe WHO has it, HOW OFTEN, and WHY existing solutions fail?)
- Evidence of demand (traction, testimonials, analogous market signals)

Calibration anchors in the prompt: *"Most adults in Kenya can't access financial services" (M-Pesa) = 10; "Teams struggle with project tasks" (Asana, crowded) = 5; "I want a workout-feed app" = 3.*

### Founder
LinkedIn-only. Rubric:
- **9–10** = Repeat founder with prior exit, OR top-tier pedigree (IIT/IIM/Stanford/MIT/Wharton/Harvard) + senior role at a top tech co + domain expertise
- **7–8** = Strong pedigree (top school OR top company) + prior startup experience OR 5+ yrs leadership + clear domain fit
- **5–6** = Mid-tier school/company, 3–5 yrs relevant work
- **3–4** = Thin credentials, recent grad, weak domain fit
- **1–2** = No verifiable background or red flags

If LinkedIn returns a login wall, HTTP 999, or empty body, the founder score is left blank (NA). The script never falls back to Google Search for founder data — it would too easily attribute the wrong person.

### Anti-bias guardrails (applied across all four scores)

The prompts include explicit instructions to avoid common evaluation biases:

1. **Independence** — score each axis separately; don't let one inflate another (halo effect is auto-flagged as a validation warning if all three business scores are identical)
2. **Sector-neutral** — hype sectors (AI, Web3) get no bonus; boring sectors (logistics, agri, MSME) get no penalty
3. **Geography-neutral** — India-only startups are scored on their actual SAM, not penalized for not being global
4. **Founder-identity-neutral** for the business scores — gender, ethnicity, school don't factor into Market/MOAT/Problem
5. **Aesthetics-neutral** — beautiful deck for a weak business is still a weak business
6. **Buzzwords = zero** — "AI-powered", "blockchain", "revolutionary" count for nothing without specific technical evidence
7. **Founder-claim skepticism** — all founder claims are hypotheses to verify, not facts
8. **Calibration** — most batches cluster at 4–6; 9–10 are rare
9. **No hallucination** — say "unable to verify" rather than invent
10. **When in doubt, round down** — pick the lower of two adjacent scores and explain the upside case

For founder scoring specifically: no name/gender/ethnicity inference, school-brand-neutral, recent-graduate cap of 5, evidence-only (no inference of unstated facts).

### Output validation

Every Gemini response is checked before being written to CSV. Warnings (not errors) are surfaced in `validation_warnings_market_moat_problem` and `validation_warnings_founder` columns. The checks include:

- All required fields present and non-empty
- All scores are integers in [1, 10] (string scores like `"8"` are auto-coerced)
- Confidence values are one of `low` / `medium` / `high`
- Problem severity / frequency match the allowed enum
- `moat_evidence` is ≥ 50 chars when `moat_score` ≥ 7 (anti-laziness)
- `demand_evidence` is ≥ 30 chars when `problem_score` ≥ 7 (anti-laziness)
- Halo-effect heuristic: all three business scores identical = flagged for review

Validation never blocks writing — the row is always persisted so you can audit suspect rows yourself.

---

## 6. Input CSV format

The script expects these columns (extra columns are ignored):

`Tracking ID`, `Full Name`, `LinkedIn`, `Startup Name`, `Sector`, `Team Size`, `Location`, `Stage`, `Business Model`, `Startup Age`, `Achievements`, `Past Accelerators`, `Problem Solving`, `Solution`, `Target Customer`, `USPs`, `Pitch Deck URL`, `Has Patents`, `Patent Details`

Most fields may be blank — the script renders missing values as `"N/A"` in the prompt.

---

## 7. Known limitations & gotchas

| Issue | Detail |
|---|---|
| **LinkedIn 999 errors** | LinkedIn aggressively blocks unauthenticated bot traffic. Expect 0–25% LinkedIn fetch success rate. For rows that fail, founder fields are NA — no Google Search fallback by design. |
| **Pitch deck hosts** | Only `conquestbits.org` URLs are downloaded. OneDrive (`1drv.ms`) and Google Drive share links are skipped (auth-gated). |
| **Free tier daily quota** | ~20 grounded requests/day on Gemini 2.5 Flash. Enable billing to process the full dataset. |
| **503 / 429 errors** | The script retries up to 5 times with exponential backoff. 503s are usually transient; 429s mean you've hit your daily quota. |
| **Schema changes** | If the script's `OUTPUT_COLUMNS` ever change, delete `evaluate_market_results.csv` before re-running to avoid header drift. |

---

## 8. Cost estimate (paid tier)

For the full 669-row dataset on **Gemini 2.5 Flash** (paid Tier 1):
- ~669 grounded calls (market + MOAT) + ~30–150 ungrounded calls (founder, depending on LinkedIn success rate)
- Model tokens: well under **$1**
- Google Search grounding: first 1,500 grounded calls free per day, then ~$35/1k
- **Total: < $10**

---

## 9. Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── evaluate_market.py            # the script
└── conquest-registrations-...csv # YOUR input (gitignored)
```

`evaluate_market_results.csv` is created when you run the script (also gitignored).
