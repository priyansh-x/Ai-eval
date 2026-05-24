# Conquest AI Evaluator

```
   ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██╗   ██╗███████╗███████╗████████╗
  ██╔════╝██╔═══██╗████╗  ██║██╔═══██╗██║   ██║██╔════╝██╔════╝╚══██╔══╝
  ██║     ██║   ██║██╔██╗ ██║██║   ██║██║   ██║█████╗  ███████╗   ██║
  ██║     ██║   ██║██║╚██╗██║██║▄▄ ██║██║   ██║██╔══╝  ╚════██║   ██║
  ╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝╚██████╔╝███████╗███████║   ██║
   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝
                       A I   E V A L U A T O R
```

A Python script that uses the **Gemini API** to evaluate startups along three dimensions:

| Score | What it measures | Inputs used |
|---|---|---|
| **Market Size** (1–10) | Realistic SAM, growth/CAGR, tailwinds, headwinds, geographic reach, timing | Founder text + pitch deck PDF (if available) + live Google Search |
| **Differentiation / MOAT** (1–10) | Tech/IP, data, network effects, brand, switching costs, regulatory, distribution, scale advantages | Same as above; cross-checks deck claims against independent research |
| **Problem Validation** (1–10) | Severity, frequency, prevalence, existing willingness-to-pay, articulation quality, demand evidence | Same as above |

All three scores are produced by **one** Gemini call per startup (with Google Search grounding) — the deck is downloaded once and read multimodally. Results are appended to a CSV one row at a time, so the script is **crash-safe and resumable**.

### Anti-bias & quality controls

The prompts include explicit guardrails: sector-neutral, geography-neutral, founder-identity-neutral, no halo effect across dimensions, buzzwords-without-evidence count as zero, and conservative defaults when data is thin. Every Gemini response is run through a **validator** that checks score type/range, required fields, confidence-enum values, evidence-substance for high scores, and a halo-effect heuristic. Any violations are written to a `validation_warnings` column in the CSV for auditing — they never block writing the row.

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

Drop your registration CSV into the project root. The script defaults to:

- **Input:**  `Conquest-Input.csv`
- **Output:** `Conquest-Output.csv`

If your file is named differently, either rename it to `Conquest-Input.csv` or use the `--csv` flag (see [§3.2](#32-other-useful-flags)).

> **Privacy note:** all `*.csv` files are `.gitignore`d on purpose — they contain founder names, emails, and phone numbers. Never commit them to a public repo.

---

## 3. Running the script

### 3.1 Safe first run — 15 startups only

For your first run (and to stay well within the free tier's daily quota), evaluate just **15 startups**:

```bash
python3 evaluate_market.py --limit 15
```

This will:
- Print a `CONQUEST AI EVALUATOR` banner and the run config (input file, output file, model, limits)
- Read the input CSV
- For each of the first 15 startups (skipping rows already evaluated in a prior run, and skipping empty rows):
  1. Download the pitch deck PDF if it's hosted on `conquestbits.org` (in-memory; not saved to disk)
  2. Call Gemini once with the deck + founder text + Google Search grounding → returns **Market + MOAT + Problem Validation** scores in one JSON
  3. Validate the response (scores in range, evidence present for high scores, halo-effect heuristic) and append to `Conquest-Output.csv`
- Show a per-startup progress block with a progress bar, scores, and an ETA estimate
- Print a clean run summary at the end

### 3.2 Other useful flags

```bash
# Skip the first N rows in the source CSV (e.g. skip the first 100, then take 15)
python3 evaluate_market.py --start 100 --limit 15

# Only evaluate rows that have a Pitch Deck URL (saves API calls)
python3 evaluate_market.py --only-with-deck --limit 15

# Point at a different input or output file
python3 evaluate_market.py --csv ~/Downloads/my-batch.csv --output my-results.csv --limit 15

# Full run (only do this on the paid tier — will hit free-tier quota almost immediately)
python3 evaluate_market.py
```

| Flag | Default | What it does |
|---|---|---|
| `--csv` | `Conquest-Input.csv` | Input CSV path |
| `--output` | `Conquest-Output.csv` | Output CSV path |
| `--limit N` | (none) | Process at most N new rows in this run |
| `--start N` | `0` | Skip the first N rows of the source CSV |
| `--only-with-deck` | off | Skip rows with no Pitch Deck URL |

### 3.3 Resuming

The script remembers which rows it has already scored (by Tracking ID, or by row index when the Tracking ID is blank). If you stop it with `Ctrl+C` and re-run, it picks up where it left off — **safe to interrupt at any time**.

To start fresh, delete the results file:

```bash
rm Conquest-Output.csv
```

---

## 4. Output

The script writes `Conquest-Output.csv` (or whatever you pass to `--output`) with these columns:

**Identifiers**
- `Tracking ID`, `Startup Name`, `Sector`, `Stage`, `Pitch Deck URL`

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

**Diagnostics**
- `deck_fetch_status` — `downloaded N bytes`, `skipped (unsupported host)`, `no url`, etc.
- `pitch_deck_accessed`, `pitch_deck_notes`, `web_sources_used`
- `validation_warnings` — pipe-separated list of any schema/quality issues with the model's output
- `evaluation_error` — populated only if a call failed after all retries

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

### Anti-bias guardrails (applied across all three scores)

The prompts include explicit instructions to avoid common evaluation biases:

1. **Independence** — score each axis separately; don't let one inflate another (halo effect is auto-flagged as a validation warning if all three scores are identical)
2. **Sector-neutral** — hype sectors (AI, Web3) get no bonus; boring sectors (logistics, agri, MSME) get no penalty
3. **Geography-neutral** — India-only startups are scored on their actual SAM, not penalized for not being global
4. **Founder-identity-neutral** — gender, ethnicity, school don't factor into the scores
5. **Aesthetics-neutral** — beautiful deck for a weak business is still a weak business
6. **Buzzwords = zero** — "AI-powered", "blockchain", "revolutionary" count for nothing without specific technical evidence
7. **Founder-claim skepticism** — all founder claims are hypotheses to verify, not facts
8. **Use the full range 1–10** — do NOT default to 5–7 for ambiguous cases. A real batch should span at least 4 different values per axis. Clustering everything at 6–7 is the #1 failure mode of automated scoring.
9. **No hallucination** — say "unable to verify" rather than invent
10. **When in doubt, round down** — pick the lower of two adjacent scores and explain the upside case
11. **Forcing function** — before committing to score X, state what X-1 and X+1 would look like for this startup. If you can't articulate both cleanly, pick the more conservative score.

Each axis (Market, MOAT, Problem Validation) has a **concrete worked example for every integer 1 → 10** in the prompt, not just band-level descriptions. This forces the model to actually pick the integer that best matches the closest example rather than defaulting to a comfortable mid-range.

### Output validation

Every Gemini response is checked before being written to CSV. Warnings (not errors) are surfaced in the `validation_warnings` column. The checks include:

- All required fields present and non-empty
- All scores are integers in [1, 10] (string scores like `"8"` are auto-coerced)
- Confidence values are one of `low` / `medium` / `high`
- Problem severity / frequency match the allowed enum
- `moat_evidence` is ≥ 50 chars when `moat_score` ≥ 7 (anti-laziness)
- `demand_evidence` is ≥ 30 chars when `problem_score` ≥ 7 (anti-laziness)
- Halo-effect heuristic: all three scores identical = flagged for review

Validation never blocks writing — the row is always persisted so you can audit suspect rows yourself.

---

## 6. Input CSV format

The script expects the standard Conquest registration export. Columns used in the prompts:

**Identifiers & founder:** `Tracking ID`, `Full Name`, `LinkedIn`, `Startup Name`

**Business profile:** `Sector`, `Team Size`, `Location`, `Stage`, `Business Model`, `Startup Age`

**Founder-declared text:** `Problem Solving`, `Solution`, `Target Customer`, `USPs`, `Achievements`, `Past Accelerators`

**Defensibility signals:** `Has Patents`, `Patent Details`, `Pitch Deck URL`, `Demo Video Link`

**Traction & capital signals:** `Has Revenue`, `MRR`, `Monthly Burn`, `Funding Status`, `Funding From`, `Grant Details`

**Self-awareness signals:** `Challenge 1` / `Challenge 1 Details`, `Challenge 2` / `Challenge 2 Details`, `Challenge 3` / `Challenge 3 Details`

Any column that's missing is rendered as `"N/A"` in the prompt — the script will not crash. Extra columns are ignored.

---

## 7. Known limitations & gotchas

| Issue | Detail |
|---|---|
| **Pitch deck hosts** | Only `conquestbits.org` URLs are downloaded. OneDrive (`1drv.ms`), Google Drive share links, `gamma.app`, etc. are skipped (auth-gated or unsupported). For those rows the score is built from text + Google Search only. |
| **Free tier daily quota** | ~20 grounded requests/day on Gemini 2.5 Flash. Enable billing to process the full dataset. |
| **503 / 429 errors** | The script retries up to 5 times with exponential backoff. 503s are usually transient; 429s mean you've hit your daily quota. |
| **Schema changes** | If the script's `OUTPUT_COLUMNS` ever change, delete `Conquest-Output.csv` before re-running to avoid header drift. |

---

## 8. Cost estimate (paid tier)

For the full 619-row dataset on **Gemini 2.5 Flash** (paid Tier 1):
- ~619 grounded calls total (one per startup — market + MOAT + problem all in one call)
- Model tokens: well under **$1**
- Google Search grounding: first 1,500 grounded calls free per day, then ~$35/1k
- **Total: ~$1–3** (well under daily grounding-free allowance)

---

## 9. Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── evaluate_market.py     # the script
└── Conquest-Input.csv     # YOUR input (gitignored)
```

`Conquest-Output.csv` is created when you run the script (also gitignored).
