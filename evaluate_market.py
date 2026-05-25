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

8. PERCENTILE CALIBRATION — against the ACTUAL Conquest pool, not abstract
   "all possible startups". This pool's realistic distribution:
     Score 1-2 ≈ ~8%  (red flags, no validation, hobby-tier ideas)
     Score 3-4 ≈ ~35% (real but clearly weak — many Pre-MVP/Ideation rows)
     Score 5   ≈ ~30% THE MEDIAN APPLICANT — a real but early Indian
                      startup with realistic but unproven ambition
     Score 6   ≈ ~15% (above median — some traction or differentiation)
     Score 7   ≈ ~8%  (genuinely strong on this axis)
     Score 8   ≈ ~3%  (excellent, would clearly stand out)
     Score 9   ≈ ~1%  (rare — must have verifiable scale evidence)
     Score 10  ≈ ~0%  (essentially never warranted in this pool)

   CRITICAL: most of this pool clusters at 4-5 (typical early-stage Indian
   bootstrapped startups). If you find yourself scoring 6-7 repeatedly, you
   are calibrating against a Silicon-Valley-scale benchmark, NOT against
   what this pool actually contains. Re-anchor downward.

   When you assign score X, you are explicitly claiming this startup is
   better than (cumulative %)-up-to-X of the actual pool on this axis.
   If you score 7, you're saying top ~12%. If 8, top ~4%. If 9, top ~1%.

9. NO HALLUCINATION. If you cannot find evidence for a claim via Google Search
   or in the deck, say so in the relevant field ("unable to verify") rather than
   invent supporting detail. Better to score lower with low confidence than to
   fabricate justification.

10. WHEN IN DOUBT, ROUND DOWN. If genuinely torn between two adjacent scores,
    pick the LOWER one and explain the upside case in the summary.

11. EXTERNALIZED FORCING FUNCTION (REQUIRED in JSON output).
    For each of the three scores you MUST output two extra fields:
      "<axis>_why_not_lower": one sentence explaining why this startup is
        better than score X-1, referencing the X-1 ANCHOR by name (dollar
        amount, moat type, or problem example from the per-integer list).
      "<axis>_why_not_higher": one sentence explaining why this startup
        falls SHORT of score X+1, referencing the X+1 ANCHOR by name.
    Generic answers ("market is smaller", "less defensible") will be
    rejected. You MUST quote a specific element of the anchor for X-1 and X+1.
    If score == 1, set "_why_not_lower" to "N/A — already at minimum".
    If score == 10, set "_why_not_higher" to "N/A — already at maximum".
    Doing this externalization is THE primary anti-clustering mechanism.

==================================================================================
POOL CONTEXT — calibrate against THIS pool, not against abstract "startups"
==================================================================================
This startup is one of 619 applicants to the Conquest competition (BITS Pilani).
The pool's actual demographics:
- 79% of teams are 1-10 people; 84% are bootstrapped or grant-funded (NOT VC-scale)
- Stage mix: 16% Ideation/Pre-MVP, 40% MVP, 32% Growth, 5% Expansion
- Only 48% have any revenue at all
- Among revenue-positive: MEDIAN MRR is ₹1L/month (~$1,200/month) — hobbyist scale
  75% of revenue-positive have <₹10L/month MRR (~$12K)
  Only ~10% have ₹10L+ MRR
- 43% claim patents but most are "filed" not "granted in active use"
- Dominant sectors: Healthcare/Digital Health, AI, Agriculture/Food, Climate,
  Consumer, Fintech, Manufacturing
- Most pitch decks are well-formatted but most underlying businesses are early
  and unproven — do not let deck polish inflate your scores

WHAT THIS MEANS for scoring:
- A typical applicant from this pool deserves a 4-5 across axes, NOT a 6-7
- Anything claiming "growth stage" with <₹5L MRR is closer to 4 than 7
- A "patent filed" claim with no granted IP is NOT a moat — at best a 4
- "We did 3 customer interviews" is weak validation — closer to 3-4 on problem
- 7+ requires concrete, verifiable evidence of either real traction (₹10L+ MRR,
  named enterprise customers, etc.) OR genuine structural defensibility
- 8+ requires the startup to be in the top ~3% of this pool — be very stingy

==================================================================================
HARD EVIDENCE GATES — apply BEFORE choosing any score
==================================================================================
These are MANDATORY ceilings. Read every gate. Apply the strictest cap that fits.

THE TWO MOST IMPORTANT RULES (READ TWICE):

RULE A — DEFAULT-DOWN ON WEAK EVIDENCE.
When evidence is weak, absent, or only claimed in the deck without
independent confirmation, the correct score is 3-4, NOT 5. Score 5 is
reserved for startups WITH real (early-but-verifiable) validation. If you
catch yourself defaulting to 5 because "nothing is clearly wrong", you are
anchoring. Drop to 3 or 4. It is FINE — even expected — to score MOAT 2-3
and Problem 3-4 for typical early-stage MVP startups. That is what most of
this pool looks like.

RULE B — CLAIMED ≠ VERIFIED.
A claim made ONLY in the pitch deck or founder text, with no independent
Google-search confirmation (press, public records, named third-party
artifact), counts as a WEAK signal and supports score 3-4 only. Examples
of weak-only claims:
  - "proprietary AI/algorithm/tech" without granted patent or technical paper
  - "trademark filed" / "patent pending" without grant
  - "in-house developed" without a named third-party customer
  - "partnerships with X" without a press release / public confirmation
  - "first-mover" / "first-of-its-kind" without independent corroboration
By contrast, INDEPENDENTLY VERIFIED evidence (e.g. a named enterprise
client confirmed by their press, a granted patent number, a regulatory
license number, a press-announced partnership) supports higher scores.
A startup with verified Amul/Mother-Dairy/DP-World-class clients in active
contract is a 6-7 on MOAT, not a 5.

ANTI-HALO TIE-BREAKER. If your three scores are about to come out
identical (X/X/X), force-differentiate: keep the strongest axis at X,
move at least one weaker axis to X-1 or X-2. Three identical scores is
almost always a tell that you are anchoring instead of evaluating.

MARKET SCORE GATES:
- market_size_score > 6 REQUIRES a cited source (specific market research firm,
  named report, or comparable-company revenue analogy) for the TAM estimate.
  Without a citation, CAP MARKET AT 6.
- market_size_score > 7 REQUIRES the startup's actual product/wedge to plausibly
  reach this TAM (not just claim it). If the wedge is narrow but the founder
  is citing the whole sector's TAM, CAP MARKET AT 7.

MOAT SCORE GATES:
- moat_score > 5 REQUIRES one of: (a) GRANTED (not filed) patent + evidence
  of its use in product, (b) named exclusive distribution contract verified
  by press / announcement, (c) demonstrated brand pull (named press +
  organic traction), (d) regulatory license number, OR (e) measurable
  network effects with cited user counts. Without (a)-(e), CAP MOAT AT 5.
- moat_score > 7 REQUIRES TWO of the above with verifiable evidence.
- "Patent pending", "patent filed", "first-mover advantage", and "proprietary
  technology" (without specifics) are NOT MOATS. They support score 4-5 at best.

PROBLEM SCORE GATES (the most-violated; apply VERY strictly):
- problem_score > 5 REQUIRES at least one of: (a) named paying customers
  with a stated count, (b) MRR specifically for THIS startup (cite the
  rupee amount), (c) cited PER-CUSTOMER existing spend on workarounds
  (e.g., "SMBs currently pay ₹15K/mo to chartered accountants"), or (d)
  signed LOIs/contracts with NAMED entities. Without (a)-(d), CAP PROBLEM AT 5.
- problem_score > 6 REQUIRES verifiable startup-specific traction: e.g.
  >₹2L MRR + customer count, OR >5 named enterprise pilots, OR a waitlist
  with cited number of signups.
- problem_score > 7 REQUIRES BOTH meaningful traction (>₹10L MRR or 100+
  paying users or major enterprise contracts) AND documented per-customer
  spend on inferior workarounds.
- Sector-wide reports ("the Indian fintech market is $50B") do NOT validate
  THIS startup's problem. They only validate the market opportunity.
- Founder's articulate problem description does NOT validate the problem.
  Articulation is necessary but NOT sufficient.

STAGE-BASED CAPS (use the Stage field, applied AFTER the above gates):
- Stage = Ideation:         ALL three scores cap at 5 (no MVP -> no validation possible)
- Stage = Pre-MVP:          problem_score and moat_score cap at 5; market_score cap at 7
- Stage = MVP without MRR:  problem_score caps at 5; moat_score caps at 5
- Stage = MVP with MRR <₹2L: problem_score caps at 6
- Stage = MVP with MRR ₹2-10L: problem_score caps at 7
- Stage = Growth without MRR or with MRR <₹5L: problem_score caps at 6
- Stage = Growth with MRR >₹10L: no automatic cap; apply rubric strictly
- Stage = Expansion: no automatic cap

EXCEPTIONS to stage caps: a startup may exceed its stage cap ONLY if it has
extraordinary verifiable evidence (e.g., granted patent already in use,
multi-crore enterprise contract signed, regulatory license already in hand).
You MUST cite the specific exception in the relevant analysis_summary field.

==================================================================================
CALIBRATION EXAMPLES — anchor your scoring to these (LLMs pattern-match better
than they instruction-follow; these are your strongest calibration signal)
==================================================================================

Reference example A — WEAK applicant (score 2):
  Profile: Solo-founder Pre-MVP app "combine my mood, dog mood, and friends mood
           into one feed". Ideation stage. No users. No IP. No revenue.
           Generic claim of "AI-powered". No customer interviews cited.
  CORRECT scoring:
    market_size_score = 2  (hyper-niche, <$10M TAM, no clear demand)
    moat_score        = 2  (nothing — solo MVP, no IP, no users, replicable in a weekend)
    problem_score     = 2  (hypothetical, no demand signal, no validation)

Reference example B — BELOW-MEDIAN applicant (scores 3-4):
  Profile: Pre-MVP D2C wellness brand. "Patent filed" (not granted).
           Claims "3 customer interviews done". No MRR. 800 Instagram followers.
           "Proprietary formulation" — not verifiable.
  CORRECT scoring:
    market_size_score = 4  (sub-billion India wellness niche)
    moat_score        = 3  (filed patent + nothing verifiable; CLAIMED ≠ VERIFIED)
    problem_score     = 4  (real problem; no concrete validation cited)

Reference example C — MEDIAN applicant, where most of this pool lives (scores 4-5):
  Profile: MVP-stage vertical SaaS for Indian SMBs in one function (HR / accounting).
           5 paying pilots stated by founder, ~₹40K MRR. No granted IP.
           Founder cites 12 customer interviews. No press coverage yet.
  CORRECT scoring:
    market_size_score = 5  (mid-size India SAM ~$1-3B)
    moat_score        = 4  (5 pilots = early signal, NOT yet a moat)
    problem_score     = 5  (5 paying pilots = real but early validation)

Reference example D — ABOVE-MEDIAN applicant (scores 6-7):
  Profile: Growth-stage fintech. Verified ₹15L/mo MRR (cited number).
           Press-announced partnership with a major NBFC (Google-verifiable).
           Cited per-customer spend on workarounds (₹15K/mo to CAs).
           Growing 18% MoM. 200+ paying SMB customers.
  CORRECT scoring:
    market_size_score = 7  (large $8-15B India SMB fintech SAM, verified)
    moat_score        = 6  (verified named partnership + early scale = real moat)
    problem_score     = 7  (₹15L MRR + cited per-customer spend + repeat behavior)

Reference example E — STRONG applicant (scores 8):
  Profile: Growth-stage deep-tech. GRANTED patent (specific patent number)
           in active use in product. Named verified enterprise contracts with
           multiple Fortune-India companies (Tata, Reliance). ₹50L+ MRR.
           Multiple bank/regulator clearances. Strong industry press coverage.
  CORRECT scoring:
    market_size_score = 8  (very large TAM, with verifiable execution path)
    moat_score        = 8  (granted IP + multiple structural moats + scale)
    problem_score     = 8  (universal pain + strong verified traction)

KEY CALIBRATION TAKEAWAY from these examples:
- Most early-stage startups in this pool look like A, B, or C (scores 2-5)
- Score 5 = median = "MVP with some real-but-early validation"
- 6+ requires verifiable evidence beyond the deck
- 8+ requires standout traction (₹50L+ MRR, granted IP, F-class contracts)
- 9-10 is essentially never warranted in this pool

If your output scores never look like A or B, you are calibrating wrong.

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

PER-INTEGER MARKET ANCHORS (memorize these — score by closest match;
each anchor is calibrated to startups actually found in this pool):

   1 = Hyper-local micro-niche, <100K addressable users globally, <$10M TAM.
       Example: a rental marketplace for one type of equipment in one
       Indian city; a paid newsletter for one ultra-niche hobby.
   2 = Very small market, $10-50M TAM, stagnant or shrinking. Example:
       a physical product for one Tier-3 demographic; a consulting service
       for one rare professional certification.
   3 = Small India market, $50-300M TAM, fragmented, low growth (<10%).
       Example: a regional component supplier to one industrial vertical;
       a hyperlocal D2C brand in one city in one category.
   4 = Sub-billion India TAM ($300M-$1B), moderate growth (10-15%).
       Example: vertical SaaS for a single Indian profession (e.g.
       dentists); specialty insurance for one professional category.
   5 = TYPICAL OF THIS POOL — India SAM $1-3B, healthy growth (12-18%).
       Example: a general D2C wellness brand; B2B SaaS for Indian SMBs
       in one function; consumer app for a defined user segment.
   6 = Above pool median — India SAM $3-8B, strong growth (15-25%) or
       major tailwinds. Example: fintech infrastructure for SMB lending;
       K-12 edtech for vernacular learners; vertical SaaS with global
       expansion optionality.
   7 = Top quartile of pool — SAM $8-20B with strong growth. REQUIRES
       evidence the startup can actually reach this market (not just
       claim it). Example: broad India SMB fintech; India digital health
       for chronic conditions at scale; agri infrastructure for crop
       value chains.
   8 = Top 10% of pool — SAM $20-50B+, hypergrowth or category-defining
       for India. REQUIRES strong evidence (cited reports + comparable
       company data + plausible execution path). Example: core India
       climate-tech infrastructure (carbon, energy storage); AI infra
       platform with global TAM.
   9 = Top 3% of pool — globally relevant ($50B+ global TAM), with India
       platform play AND verifiable execution path. RARE. Example:
       global SaaS-from-India platform with US/EU expansion already
       happening; deep tech with verified global applicability.
  10 = Once-a-decade global inflection ($200B+ global TAM). ESSENTIALLY
       NEVER WARRANTED IN THIS POOL. If considering 10, default to 8 or 9.

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

PER-INTEGER MOAT ANCHORS (memorize these — score by closest match;
each anchor is calibrated to startups actually found in this pool):

   1 = No moat at all; pure copy of incumbents. Example: yet another
       Instagram-store reselling generic apparel; me-too service in a
       saturated market with no differentiation.
   2 = Feature, not a company; replicable in a weekend by a competent
       team. Example: thin UI wrapper around an LLM API with no
       fine-tuning, no proprietary data, no exclusive distribution;
       another no-code form builder; solo-founder MVP with literally
       nothing but an idea.
   3 = MOST COMMON FOR EARLY-STAGE in this pool: an MVP/Pre-MVP with
       claimed differentiation ("proprietary tech", "first-mover",
       "filed a patent") but NOTHING independently verifiable.
       Positioning-only (cheaper, niche-focus); no IP, no lock-in, no
       network effects. Example: "WhatsApp for doctors" me-too; B2B
       SaaS with claimed proprietary algorithms but no granted IP;
       D2C brand with <10K followers and no press.
   4 = Light execution/brand differentiation with at least ONE small
       verifiable signal (a named pilot customer, modest organic
       traction, or a clearly filed-but-unverified patent in a deep-tech
       space). Well-funded competitor could erode in <1 year. Example:
       regional D2C brand with one named retail listing; B2B SaaS with
       one paying pilot logo; deep-tech with a publicly-visible patent
       filing in a real technical area.
   5 = ONE early-stage moat WITH REAL VERIFIABLE BASIS — a meaningful
       nascent brand (>50K real organic followers + press mentions) OR
       multiple named pilot customers OR small but measurable network
       effect (e.g. monetized two-sided community) OR a granted (not
       pending) patent in a technical area. This requires INDEPENDENT
       evidence, not just deck claims. A startup with only "in-house
       tech" + "filed patent" + no verified clients is NOT a 5 — it's
       a 3 or 4.
   6 = Above pool median — ONE genuinely strong moat WITH EVIDENCE:
       granted patent IN ACTIVE USE in product, OR demonstrated brand
       pull with paying customers, OR named distribution exclusivity.
       Example: deep-tech with one granted patent driving real cost/
       quality advantage; fintech with verified exclusive bank deal.
   7 = Top quartile — TWO real moats compounding (e.g. tech IP +
       distribution; data + brand; network effects nascent + switching
       costs). REQUIRES verifiable evidence of both. Example: vertical
       SaaS deeply embedded in customer workflows WITH proprietary
       data accumulating.
   8 = Top 10% — structural moat ALREADY COMPOUNDING — verified
       two-sided network effects, regulatory license hard to obtain
       (banking, healthcare, defense), or unique data lead at scale.
       REQUIRES concrete evidence (license number, active user counts,
       data scale figures). Example: neobank with banking license +
       10K+ active users; healthtech with regulatory clearance + scale.
   9 = Top 3% — multiple compounding structural moats; clear category
       leadership emerging within India. RARE in this pool. Requires
       verifiable scale (e.g. >$1M ARR + multiple structural moats).
  10 = Category-defining defensibility at global scale (planetary data
       lead, regulatory monopoly, billion-user network effects).
       ESSENTIALLY NEVER WARRANTED IN THIS POOL. Default to 8 or 9.

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

PER-INTEGER PROBLEM ANCHORS (memorize these — score by closest match;
each anchor is calibrated to startups actually found in this pool):

   1 = Solution looking for a problem; founder can't articulate WHO has
       it, WHEN, or WHY. Example: "an app that combines 3 random things
       no one asked for"; vague TG like "for everyone".
   2 = Hypothetical / preachy ("people should X"); zero evidence of
       demand or spend; no validation conversations cited. Example:
       "people should drink more water" — true, but nobody pays.
   3 = Real but minor; existing free tools or habits cover it adequately;
       few would actively pay. Example: another mood-journal app;
       fancier reminder tool in a saturated category.
   4 = MOST COMMON FOR EARLY-STAGE in this pool: a real problem with
       a definable user, but no concrete validation evidence — no
       interviews cited, no waitlist size, no MRR, no named pilot.
       Pre-MVP/Ideation stage rows almost always belong here unless
       they have something extraordinary. Example: "small teams need
       a single project tool" when Notion/Trello are good enough; an
       MVP-stage startup with founder narrative but zero traction.
   5 = Real problem WITH some genuine validation: founder cites specific
       interview counts, has a waitlist with a stated number (>50), or
       has 1-4 named pilot customers (no MRR yet). Not just "we talked
       to users" but actual evidence the validation happened. If you
       can't find a specific number / name in the founder text or deck,
       this is NOT a 5 — it's a 4.
   6 = Above pool median — clear painful problem + customers ARE paying
       for inferior workarounds today + EARLY paying traction (e.g.
       <₹2L MRR with growing customers, or 5+ named pilot customers).
       Example: "Indian SMBs paying CAs ₹15K/mo for monthly compliance
       when software could automate it" + 5 paying pilots.
   7 = Top quartile — SEVERE + FREQUENT pain WITH ALL THREE:
       (i) >₹2L MRR growing OR >5 named paying customers
       (ii) cited PER-CUSTOMER existing spend on workarounds (specific
            rupee amount, NOT sector-wide market data)
       (iii) repeat-customer behavior or strong retention signal
       Industry reports alone do NOT qualify. Example: "Indian gig
       workers have no access to instant healthcare financing" + ₹3L MRR
       from 200 customers + 20% MoM + cited current spend of ₹5K/yr
       per worker on unsecured loans.
   8 = Top 10% — universal pain + STRONG TRACTION. REQUIRES e.g.
       >₹10L/mo MRR, OR national-scale partnerships with named entities,
       OR multiple signed enterprise contracts. The startup-specific
       traction must be verifiable; sector size does NOT substitute.
   9 = Top 3% — painfully obvious-in-hindsight + near-immediate demand
       once solution ships + significant scale already proven (e.g.
       ₹50L+ MRR, OR 10K+ active paying users). RARE in this pool.
  10 = Generational-level pain; absence of solution is itself a crisis;
       users will do anything to solve it. Reserve for "this is THE
       story of the decade" — ESSENTIALLY NEVER WARRANTED IN THIS POOL.

==================================================================================
OUTPUT
==================================================================================
Return ONLY a raw JSON object (no markdown fences, no prose outside JSON), in
EXACTLY this schema. All string values MUST be single-line (no newlines inside strings).
All scores MUST be integers in [1, 10]. All confidence values MUST be one of
"low", "medium", "high".

{{
  "market_size_score": 5,
  "market_why_not_lower": "A score of 4 would mean sub-$1B TAM. This startup is BETTER than a 4 because [reference the score-X-1 anchor with a concrete differentiator].",
  "market_why_not_higher": "A score of 6 would mean ~$3-5B SAM with 12-18% growth. This startup falls SHORT of a 6 because [reference the score-X+1 anchor with a concrete deficit].",
  "calculated_tam": "Your INDEPENDENT TAM, e.g. '$12B SAM by 2030 (India)' + 1-line basis",
  "deck_tam_claim": "What the deck/founder claimed, or 'not stated' / 'not available'",
  "cagr": "Your researched CAGR + year range, e.g. '18% (2024-2030)', or 'unknown'",
  "growth_tailwinds": "1-2 key tailwinds, single line",
  "growth_headwinds": "1-2 key headwinds, single line",
  "geographic_scope": "e.g. 'India-only', 'India + SEA', 'Global'",
  "market_analysis_summary": "3-5 sentences integrating evidence, explaining why this exact score and not one notch higher or lower.",
  "market_confidence": "low | medium | high",

  "moat_score": 5,
  "moat_why_not_lower": "A score of 4 would mean only execution/brand differentiation with no structural moat. This startup is BETTER than a 4 because [reference the score-X-1 anchor].",
  "moat_why_not_higher": "A score of 6 would mean ONE genuinely strong moat (distribution exclusivity, real brand pull) or two stacking weak ones. This startup falls SHORT because [reference the score-X+1 anchor].",
  "moat_types_present": "Comma list of moats GENUINELY present with evidence, e.g. 'Tech IP, Distribution'. Empty string if none.",
  "deck_moat_claim": "What the deck/founder claimed as their moat/USP, or 'not stated'",
  "moat_evidence": "Specific evidence for the moats listed (e.g. 'Patent US-12345 granted 2023'). If moat_score >= 7, this MUST be substantive (50+ chars).",
  "moat_risks": "What would erode the moats — 1-2 sentences",
  "moat_analysis_summary": "3-5 sentences justifying this exact score. Single line.",
  "moat_confidence": "low | medium | high",

  "problem_score": 5,
  "problem_why_not_lower": "A score of 4 would mean a real problem but with acceptable workarounds. This startup is BETTER than a 4 because [reference the score-X-1 anchor].",
  "problem_why_not_higher": "A score of 6 would mean strong validation + customers already paying for inferior solutions. This startup falls SHORT because [reference the score-X+1 anchor].",
  "problem_severity": "Pick EXACTLY ONE: high | medium | low. Do NOT hyphenate or combine (e.g. 'medium-high' is invalid).",
  "problem_frequency": "Pick EXACTLY ONE: daily | weekly | monthly | yearly | continuous | one-time | unclear. If pain is multi-frequency, pick the most relevant single value. Do NOT combine values.",
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
ALLOWED_FREQUENCY = {"daily", "weekly", "monthly", "yearly", "continuous", "one-time", "unclear"}


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

    # Externalized forcing function: each axis must explain why-not-lower and
    # why-not-higher to prove the model actually did the comparative reasoning.
    axis_fields = (
        ("market_size_score", "market_why_not_lower", "market_why_not_higher"),
        ("moat_score",         "moat_why_not_lower",   "moat_why_not_higher"),
        ("problem_score",      "problem_why_not_lower", "problem_why_not_higher"),
    )
    for score_field, low_field, high_field in axis_fields:
        score = result.get(score_field)
        if isinstance(score, int):
            if score > 1:
                _check_substantive(result.get(low_field),
                                   f"{low_field} (required when score>1)", 40, errors)
            if score < 10:
                _check_substantive(result.get(high_field),
                                   f"{high_field} (required when score<10)", 40, errors)

    # Halo-effect check: warn (not error) if all three scores collapse to the same value
    scores = [result.get(f) for f in ("market_size_score", "moat_score", "problem_score")]
    if all(isinstance(s, int) for s in scores) and len(set(scores)) == 1:
        errors.append(f"halo-effect suspect: all three scores identical ({scores[0]}) — review prompts/output")

    # Soft clustering check: warn if all three scores fall in {6, 7, 8} —
    # the failure mode where the model defaults to "polite mid-high"
    if all(isinstance(s, int) for s in scores) and all(6 <= s <= 8 for s in scores):
        errors.append(f"score-cluster suspect: all three scores in 6-8 ({scores}) — model may be anchoring to mid-high default")

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
    # Lower temperature pushes the model to follow the explicit rubric/evidence
    # gates more strictly instead of drifting to a "polite mid-high" default.
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.3,
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
    "market_why_not_lower",
    "market_why_not_higher",
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
    "moat_why_not_lower",
    "moat_why_not_higher",
    "moat_types_present",
    "deck_moat_claim",
    "moat_evidence",
    "moat_risks",
    "moat_confidence",
    "moat_analysis_summary",
    # problem validation
    "problem_score",
    "problem_why_not_lower",
    "problem_why_not_higher",
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


def normalize_scores(results_path: Path) -> None:
    """Post-hoc rank normalization: re-bin raw scores against the cohort so the
    final scores span 1-10 with a realistic bell-curve distribution.

    Why this exists: even with strict prompts + low temperature + few-shot
    examples, Gemini compresses absolute scores toward the middle (the
    irreducible "be nice" RLHF bias). Rank normalization solves this by
    GUARANTEE: the relative ordering the model produces is preserved, but the
    final scores are spread across 1-10 with the expected distribution.

    Target distribution (matches what a real 600-row competition pool looks like):
       1: 3%   2: 7%    3: 15%   4: 20%   5: 20%
       6: 15%  7: 10%   8: 6%    9: 3%    10: 1%

    Adds three columns: market_size_score_normalized, moat_score_normalized,
    problem_score_normalized. The original *_score columns are untouched.
    """
    if not results_path.exists():
        print(f"Cannot normalize: {results_path} does not exist.")
        return

    df = pd.read_csv(results_path)
    n = len(df)
    if n == 0:
        print("Cannot normalize: empty results file.")
        return

    # Cumulative bucket edges (% of pool that falls AT OR BELOW each integer score)
    cum_pct = {1: 0.03, 2: 0.10, 3: 0.25, 4: 0.45, 5: 0.65,
               6: 0.80, 7: 0.90, 8: 0.96, 9: 0.99, 10: 1.00}

    def percentile_to_score(pct: float) -> int:
        for score in range(1, 11):
            if pct <= cum_pct[score]:
                return score
        return 10

    for axis in ("market_size_score", "moat_score", "problem_score"):
        if axis not in df.columns:
            continue
        # Rank ascending; ties get average rank. Convert to fractional percentile.
        # method='average' so ties don't create artificial spread.
        ranks = df[axis].rank(method="average", na_option="keep")
        pcts = ranks / n  # 0 < pct <= 1
        normalized = pcts.apply(lambda p: percentile_to_score(p) if pd.notna(p) else None)
        df[axis + "_normalized"] = normalized.astype("Int64")

    df.to_csv(results_path, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")

    print()
    print("=" * 80)
    print(f"  RANK-NORMALIZED — {n} startups against each other")
    print("=" * 80)
    import collections
    for axis in ("market_size_score", "moat_score", "problem_score"):
        if axis + "_normalized" not in df.columns:
            continue
        raw = df[axis].dropna().astype(int).tolist()
        norm = df[axis + "_normalized"].dropna().astype(int).tolist()
        raw_c = collections.Counter(raw)
        norm_c = collections.Counter(norm)
        raw_dist = " ".join(f"{v}:{raw_c.get(v,0)}" for v in range(1, 11))
        norm_dist = " ".join(f"{v}:{norm_c.get(v,0)}" for v in range(1, 11))
        print(f"  {axis.replace('_score', ''):14}")
        print(f"     raw  {raw_dist}")
        print(f"     norm {norm_dist}")
    print("=" * 80)
    print(f"  Wrote *_normalized columns to {results_path.name}")


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
    parser.add_argument("--normalize", action="store_true",
                        help="After scoring (or standalone if --limit 0), rank-normalize "
                             "scores across the cohort to guarantee a 1-10 spread. "
                             "Adds *_normalized columns; raw scores unchanged.")
    parser.add_argument("--normalize-only", action="store_true",
                        help="Skip scoring entirely; just rank-normalize the existing "
                             "output CSV and exit. Useful for re-binning after each batch.")
    args = parser.parse_args()

    # Apply CLI overrides for input/output paths
    csv_path = Path(args.csv).expanduser().resolve()
    RESULTS_PATH = Path(args.output).expanduser().resolve()

    # --normalize-only short-circuits everything else: rank-normalize the
    # existing output and exit. No Gemini calls. Useful for re-binning after
    # each incremental batch.
    if args.normalize_only:
        normalize_scores(RESULTS_PATH)
        return

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
                "market_why_not_lower": result.get("market_why_not_lower"),
                "market_why_not_higher": result.get("market_why_not_higher"),
                "calculated_tam": result.get("calculated_tam"),
                "deck_tam_claim": result.get("deck_tam_claim"),
                "cagr": result.get("cagr"),
                "geographic_scope": result.get("geographic_scope"),
                "growth_tailwinds": result.get("growth_tailwinds"),
                "growth_headwinds": result.get("growth_headwinds"),
                "market_confidence": result.get("market_confidence"),
                "market_analysis_summary": result.get("market_analysis_summary"),
                "moat_score": result.get("moat_score"),
                "moat_why_not_lower": result.get("moat_why_not_lower"),
                "moat_why_not_higher": result.get("moat_why_not_higher"),
                "moat_types_present": result.get("moat_types_present"),
                "deck_moat_claim": result.get("deck_moat_claim"),
                "moat_evidence": result.get("moat_evidence"),
                "moat_risks": result.get("moat_risks"),
                "moat_confidence": result.get("moat_confidence"),
                "moat_analysis_summary": result.get("moat_analysis_summary"),
                "problem_score": result.get("problem_score"),
                "problem_why_not_lower": result.get("problem_why_not_lower"),
                "problem_why_not_higher": result.get("problem_why_not_higher"),
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

    # Auto rank-normalize at end of run if requested. This re-bins the entire
    # output CSV (raw scores untouched) so the *_normalized columns span 1-10.
    if args.normalize:
        normalize_scores(RESULTS_PATH)


if __name__ == "__main__":
    main()
