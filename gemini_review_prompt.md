# GEMINI FULL REVIEW PROMPT — Adaptive Racial Fairness Framework

> **Instructions:** Paste this entire document into Gemini. It contains the complete project history, all code, all documentation, validation results, and strategic context. The ask is at the bottom.

---

## PART 1: PROJECT OVERVIEW

**Project:** Adaptive Racial Fairness Framework
**Creator:** Matt Jackson (Detroit, MI)
**Started:** June 2025
**Current Date:** March 15, 2026
**Repository:** https://github.com/[repo] (private)
**Deployed API:** https://adaptive-racial-fairness-framework.onrender.com
**Built with:** Claude Code (Anthropic Claude Opus/Sonnet)

### Core Claim
A fairness audit is only valid if the community subject to the decision system defined the fairness standard. Current frameworks (AIF360, Fairlearn, NYC LL144) let researchers or regulators set thresholds. This framework lets affected communities set them, and proves empirically that who defines fairness changes the outcome of the audit.

### What Makes This Different From Everything Else
Every existing algorithmic fairness tool treats the fairness definition as a researcher input. This framework treats it as a community governance output — with provenance tracking (UUID, timestamp, participant count, facilitator, decision method, dissent record). The result is that audits are classified as either "community_valid" or "standard" based on whether a real community session produced the configuration.

---

## PART 2: COMPLETE GIT HISTORY (Chronological)

```
ebee909 2025-06-04  Initial commit for Adaptive Racial Fairness Framework
171cac5 2025-06-04  Initial commit (.gitignore, LICENSE, README)
87a25c5 2025-06-04  Updated deploy_dash_app.py with new fairness reweighting logic
10823e7 2025-06-04  Update 06/04/25 (styles, dashboard improvements)
b9a9e6e 2025-06-04  Callback errors and improve data handling
fecd8ac 2025-06-27  Final debug pass - pausing Dash, moving toward refactor
98e4942 2025-06-30  Updated README
428f965 2026-03-10  Add FastAPI audit API, CSV upload dashboard, and fairness improvements
305bfbe 2026-03-10  Add Render deployment config
036c171 2026-03-10  Fix Render build: pin Python 3.11 and scipy 1.11.4
2b041bc 2026-03-10  Add landing page with live API demo
a261ab1 2026-03-10  Add CORS middleware
3376903 2026-03-11  Fix CORS: allow OPTIONS preflight through auth middleware
2b70dd1 2026-03-11  Fix reference group: default to White, fall back if not in dataset
101bb56 2026-03-13  Fix retry error handling: wrap second fetchAudit in try/catch
a85adaa 2026-03-13  Add framework documentation, community input layer, PDF reports, remediation
6215f5c 2026-03-13  Wire adversarial debiasing to API as /audit/debias endpoint
c6983fd 2026-03-14  Fix 6 critical bugs found in Opus deep audit
44c2ebb 2026-03-14  Add HMDA lending and COMPAS recidivism datasets
2c0f20a 2026-03-14  Add HMDA Michigan lending dataset (4,463 records)
```

**Key inflection points:**
- Jun 2025: Basic Dash dashboard with hardcoded data
- Mar 10, 2026: Rebuilt as FastAPI API + deployed to Render + landing page
- Mar 13, 2026: Full documentation suite, community input layer, PDF reports, adversarial debiasing
- Mar 14, 2026: 6 critical bug fixes, multi-domain validation data (HMDA + COMPAS)
- Mar 15, 2026: Validation study, threshold sensitivity analysis, reproducibility runner, CDF v1.0 spec, integration adapters, config registry

---

## PART 3: ARCHITECTURE

### Three-Layer System

**Layer 1 — Governance (Community Input → Configuration → Traceability)**
- `community_input.py`: `build_community_config()` creates configs with UUID provenance
- `specs/community_fairness_config_v1.schema.json`: JSON Schema for the config format
- `specs/CDF_SPECIFICATION_v1.md`: Formal standard document
- `registry/`: Public registry for community-published configs

**Layer 2 — Audit (Data Ingestion → Metric Computation → Report)**
- `racial_bias_score.py`: Per-group mean outcomes + disparity score
- `fairness_audit.py`: Group outcomes and disparate impact computation
- `report_generator.py`: Professional PDF audit reports via ReportLab
- `api/main.py`: FastAPI with 9 endpoints

**Layer 3 — Mitigation (Reweighting → Adversarial Debiasing → Validation)**
- `fairness_reweight.py`: Per-sample reweighting aligned to community targets
- `adversarial_fairlearn.py`: ExponentiatedGradient pipeline with pre/post comparison

### API Surface (9 endpoints)
- `GET /health`
- `POST /audit` — JSON audit
- `POST /audit/csv` — CSV upload audit
- `POST /audit/pdf` — CSV upload → professional PDF report download
- `POST /audit/remediate` — full loop: audit → reweight → pre/post DI comparison
- `POST /audit/debias` — adversarial debiasing via ExponentiatedGradient
- `POST /audit/compliance` — check any dataset against any published CDF v1.0 community config
- `POST /reweight` — JSON reweight
- `POST /reweight/csv` — CSV reweight

### Integration Adapters
- `integrations/aif360_adapter.py`: Wraps IBM AIF360 with community config governance
- `integrations/fairlearn_adapter.py`: Wraps Microsoft Fairlearn with community-defined constraints

---

## PART 4: CORE ALGORITHMS

### Disparate Impact Ratio
```
DI(g) = outcome_rate(g) / outcome_rate(reference_group)
```
Where reference_group defaults to White (EEOC standard) but can be community-defined.

### Community-Valid Audit Classification
An audit is classified as "community_valid" if and only if:
1. A community fairness configuration exists
2. The configuration has valid provenance (UUID, date, participants ≥ 1, input_protocol specified)
3. The configuration is not stale (< 365 days old)

Otherwise, the audit is classified as "standard" (researcher defaults applied).

### Reweighting Algorithm
For each sample in priority group g:
- If favorable outcome: `weight = target_rate / group_rate`
- If unfavorable outcome: `weight = (1 - target_rate) / (1 - group_rate)`

Where `target_rate` is the outcome rate of the community-defined reference group.

---

## PART 5: EMPIRICAL VALIDATION RESULTS

### Validation Study Output (3 datasets)

**Dataset 1: HR Hiring**
- Records: 8 | Groups: 3
- Results: Latinx flagged at both θ=0.8 and θ=0.9
- No difference between default and community thresholds (too small a dataset)

**Dataset 2: HMDA Lending (Michigan)**
- Records: 3,831 | Groups: 5
- Source: CFPB/HMDA federal data

| Group | Rate | DI | Default (0.8) | Community (0.9) |
|-------|------|-----|---------------|-----------------|
| American Indian or Alaska Native | 13.8% | 0.3283 | FLAGGED | FLAGGED |
| Asian | 32.3% | 0.7698 | FLAGGED | FLAGGED |
| Black or African American | 34.3% | 0.8174 | ok | **FLAGGED** |
| Native Hawaiian/Pacific Islander | 20.0% | 0.4762 | FLAGGED | FLAGGED |
| White | 42.0% | 1.0000 | ok | ok |

**KEY FINDING:** Black or African American mortgage applicants have DI = 0.8174. This is 0.0174 above the EEOC threshold of 0.80. They PASS the federal standard but FAIL at θ=0.85 or above. A community-defined threshold catches this; the default does not.

**Dataset 3: COMPAS Recidivism**
- Records: 6,837 | Groups: 5
- Source: ProPublica

| Group | Rate | DI | Default (0.8) | Community (0.9) |
|-------|------|-----|---------------|-----------------|
| African-American | 48.6% | 0.8010 | ok | **FLAGGED** |
| Asian | 71.9% | 1.1854 | ok | ok |
| Caucasian | 60.6% | 1.0000 | ok | ok |
| Hispanic | 63.6% | 1.0485 | ok | ok |
| Native American | 44.4% | 0.7328 | FLAGGED | FLAGGED |

**KEY FINDING:** African-American defendants have DI = 0.8010. They are 0.0010 above the EEOC threshold. Same pattern as HMDA — barely passing, threshold-dependent.

### Threshold Sensitivity Sweep (θ = 0.70 to 0.95)

Both Black/African American groups in HMDA (DI=0.8174) and COMPAS (DI=0.8010) are threshold-dependent. They pass at θ=0.80 but fail at θ=0.85. The 0.80 threshold is not a neutral scientific fact — it's a policy choice that actively determines which disparities are visible.

### Reproducibility
- `reproduce.py` runs 22 verification checks
- All 22 pass
- Any independent researcher can clone the repo and reproduce all findings

---

## PART 6: DOCUMENTATION SUITE

### 1. Problem Specification (`docs/problem_specification.md`)
Formal problem statement. Core falsifiable claim: "A fairness audit is only valid if the community defined the standard." Three compounding failures identified: measurement mismatch, adoption failure, accountability gap.

### 2. Literature Gap Analysis (`docs/literature_gap.md`)
Three-generation survey: metric definition (2016-2019), toolkits/mitigation (2018-2022), governance/accountability (2021-present). Maps what AIF360, Fairlearn, NYC LL144, and participatory ML literature contribute versus miss. Precise gap: no existing system gives affected communities a formal role in defining the fairness target as a required structural input to the audit.

### 3. Algorithm Specification (`docs/algorithm_specification.md`)
Full mathematical formalization — notation, DI formula, reweighting algorithm, community-valid audit definition, output specification. O(n) computational complexity.

### 4. System Architecture (`docs/system_architecture.md`)
Three-layer architecture (Governance → Audit → Mitigation). Data flow diagrams, deployment architecture, gap map.

### 5. Validation Methodology (`docs/validation_methodology.md`)
Three validation dimensions: technical (unit tests, EEOC consistency), construct (does CDF capture what communities mean?), consequential (does it produce better outcomes?). Publication readiness checklist.

### 6. Community Input Protocol (`docs/community_input_protocol.md`)
Complete facilitation guide. Who's in the room (10+ participants, demographic diversity requirements), what they see (data summary sheet, threshold explainer), 90-minute session structure (4 parts), decision methods (consensus/supermajority/median), post-session configuration creation, ethical considerations, adaptations for city government/corporate/academic contexts.

### 7. Preprint Draft (`docs/preprint_draft.md`)
arXiv/FAccT paper skeleton. 8 sections. Title: "Community-Defined Fairness: An Adaptive Framework for Participatory Algorithmic Auditing." Three claimed contributions. Section 6 placeholder for community session results.

---

## PART 7: CDF v1.0 STANDARD (Community-Defined Fairness)

The strategic play: separate the standard from the tool. If other tools adopt the format, the format becomes the standard.

### What CDF v1.0 defines:
- **Community Fairness Configuration (CFC):** A JSON document containing priority_groups, fairness_target, fairness_threshold, and provenance
- **Provenance record:** UUID, input_protocol, input_date, input_participants, facilitator, decision_method, dissent record
- **Audit classification:** community_valid vs. standard
- **Conformance requirements:** What producers, consumers, and registries must do
- **Interoperability mapping:** How to translate CFC into AIF360 and Fairlearn parameters
- **Versioning:** Semantic versioning, backward compatibility rules

### Integration adapters built:
- AIF360 adapter: `CommunityAIF360Audit` class — loads community config, runs AIF360 audit with community parameters, falls back to pure pandas if AIF360 not installed
- Fairlearn adapter: `CommunityFairlearnMitigation` class — loads community config, runs Fairlearn mitigation with community-defined constraint parameters

### Config Registry:
- Organized by jurisdiction/domain (e.g., `registry/michigan/lending/detroit_2026.json`)
- Machine-readable index (`registry/index.json`)
- Submission process: run session → build config → validate → place in directory → update index → PR
- Currently empty — awaiting first real community session

---

## PART 8: WHAT HAS NOT BEEN DONE YET

1. **No real community session has been conducted.** The "community-defined" threshold of 0.9 used in the validation study was chosen by the researcher, not a community. This is the single biggest gap.

2. **The HR dataset is 8 records.** Too small for publication. The HMDA (3,831) and COMPAS (6,837) datasets are sufficient.

3. **No comprehensive test suite.** `tests/test_fairness_pipeline.py` exists but is empty. No pytest tests written.

4. **No academic co-author.** The documentation is publication-grade but has no institutional backing.

5. **No real community config in the registry.** The registry infrastructure exists but is empty.

6. **Preprint not submitted.** The skeleton exists but Section 6 (community session results) is a placeholder.

---

## PART 9: COMPETITIVE LANDSCAPE

| Framework | Metrics | Mitigation | Community Input | Provenance | Config Standard |
|-----------|---------|------------|-----------------|------------|-----------------|
| AIF360 (IBM) | ✅ 70+ metrics | ✅ Pre/in/post | ❌ | ❌ | ❌ |
| Fairlearn (Microsoft) | ✅ | ✅ ExponentiatedGradient | ❌ | ❌ | ❌ |
| What-If Tool (Google) | ✅ Visual | ❌ | ❌ | ❌ | ❌ |
| NYC LL144 | DI only | ❌ | ❌ Auditor discretion | ❌ | ❌ |
| EU AI Act | Defines categories | ❌ | ❌ Regulatory | ❌ | ❌ |
| Colorado AI Act | "Reasonable care" | ❌ | ❌ Undefined | ❌ | ❌ |
| **This Framework** | ✅ DI, SP, Disparity | ✅ Reweight + Adversarial | ✅ Full protocol | ✅ UUID + provenance | ✅ CDF v1.0 |

**The gap:** No existing system gives affected communities a formal, traceable role in defining the fairness standard used in an algorithmic audit. This framework does.

---

## PART 10: REGULATORY TIMING

- **Colorado AI Act:** Goes live June 2026. Requires "reasonable care" to prevent algorithmic discrimination. Does not define what "reasonable care" means. This framework could define it.
- **EU AI Act:** High-risk AI systems require conformity assessments. No standard for community involvement.
- **NYC Local Law 144:** Requires annual bias audits of automated employment decision tools. Auditors have full discretion over methodology. No community input required.
- **Michigan:** HB 4667 and related AI bills under consideration. Michigan AG Dana Nessel active on algorithmic discrimination.
- **HMDA finding:** Black Michigan mortgage applicants at DI = 0.8174 — directly relevant to Michigan state policy.

---

## THE ASK

You are Gemini. You have the complete project — every line of code, every document, every finding, the full git history, and the strategic context.

**Please do the following:**

### 1. VERIFY THE RESEARCH
- Is the core claim ("who defines fairness changes the outcome") supported by the empirical evidence?
- Is the HMDA finding (Black borrowers DI=0.8174, passing at 0.80 but failing at 0.85+) correctly computed and significant?
- Is the literature gap real? Has anyone else built a working system that accepts community-defined fairness parameters with provenance tracking?
- Are there methodological weaknesses that would prevent publication at FAccT, AIES, or similar venues?
- What would a peer reviewer's strongest objection be?

### 2. IDENTIFY WHAT'S MISSING
- What technical gaps exist in the codebase?
- What documentation gaps exist?
- What would a patent examiner need that isn't here?
- What would a Nobel committee (or equivalent: AAAI Award, Turing Award, MacArthur Fellow) need to see beyond what's here?

### 3. PRODUCE A FORWARD PROMPT
Generate a detailed, actionable prompt that can be given back to Claude Code (Anthropic Claude Opus) to push this framework to the next level. This prompt should:

- Address every gap you identified
- Include specific technical tasks (code to write, tests to create, documentation to complete)
- Include a patent claim structure
- Include a publication strategy (where to submit, in what order)
- Include an adoption strategy (how to get from one tool to a standard)
- Be ruthlessly honest about what's strong and what's weak
- Treat this as if it's being prepared for the highest possible level of impact — not just "good enough" but "field-defining"

### 4. RATE THE CURRENT STATE
On a scale of 1-10, rate the framework on:
- Technical completeness
- Research rigor
- Novelty of contribution
- Publication readiness
- Patent viability
- Potential for real-world adoption
- Distance from "field-defining" impact

Be brutally honest. We'd rather know the truth than hear encouragement.
