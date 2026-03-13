# Validation Methodology
## Adaptive Racial Fairness Framework

**Version:** 0.1 (Draft)
**Status:** Internal — not for distribution

---

## 1. Purpose

This document defines how we prove the framework works. Validation has three distinct dimensions:

1. **Technical validity** — does the framework compute what it claims to compute, correctly?
2. **Construct validity** — does "community-defined fairness" actually capture what communities mean by fairness?
3. **Consequential validity** — does using this framework produce better outcomes for affected communities than not using it?

Most fairness tools only address dimension 1. This framework must address all three to support the core claim.

---

## 2. Technical Validation

### 2.1 Unit Tests

Each core function must have deterministic, reproducible test cases:

| Function | Test Case | Expected Output |
|---|---|---|
| `group_outcomes_by_race()` | Balanced dataset, equal rates | DI = 1.0 for all groups |
| `group_outcomes_by_race()` | Black rate=0.4, White rate=0.8 | DI(Black, White) = 0.5, flagged |
| `disparate_impact()` | DI = 0.79 | Flag = True (below θ=0.8) |
| `disparate_impact()` | DI = 0.80 | Flag = False (at threshold) |
| `calculate_racial_bias_score()` | All groups equal | DS = 0.0 |
| `calculate_racial_bias_score()` | Max=0.9, Min=0.1 | DS = 0.8 |
| `reweight_samples_with_community()` | Priority group rate < reference | Weights > 1 for favorable outcomes |
| Reference group selection | "White" present in dataset | r = "White" |
| Reference group selection | "White" absent, Asian=highest | r = "Asian" |
| Community config load | Valid JSON | Returns (G, r, θ) correctly |
| Community config load | Missing file | Returns defaults |

**Status:** Partial. `test_racial_score.py` and `tests/test_fairness_pipeline.py` cover some of these. Full coverage requires expansion.

### 2.2 EEOC Consistency Test

The framework must produce results consistent with the EEOC Uniform Guidelines on Employee Selection Procedures (1978) 4/5ths rule on known test cases:

- Use published EEOC enforcement case data where available
- DI < 0.8 must always be flagged when θ = 0.8
- Results must match manual calculation on the same data

### 2.3 Regression Suite

Any code change must not alter outputs on the reference dataset (`data/real_hr_data.csv`). Baseline outputs are pinned and checked on every commit.

---

## 3. Construct Validity

### 3.1 The Core Question

Does the community configuration actually reflect what the community means by "fair"? This cannot be answered by code alone — it requires a study.

### 3.2 Validation Protocol

**Phase 1 — Configuration Elicitation**
Run two parallel configuration processes on the same community:
- Process A: researcher-defined defaults (current behavior)
- Process B: structured community input session (new behavior)

Record both configurations.

**Phase 2 — Divergence Measurement**
For the same dataset, run audits using both configurations. Measure:
- Do priority groups differ? (Which groups the community says matter vs. which the researcher assumed)
- Does the reference group differ?
- Does the threshold differ?
- Do the audit outcomes (flagged/not flagged) differ?

**Hypothesis:** Community-defined configurations will differ from researcher defaults in a statistically significant proportion of cases, and the differences will cluster around historically underserved groups being added as priority groups at higher thresholds.

**Phase 3 — Community Verification**
Present the audit results from both configurations to community representatives. Ask:
- Which result better reflects what you mean by fair?
- Are there outcomes flagged by your configuration that the researcher configuration missed?

**Success condition:** Community representatives prefer the community-defined configuration output in ≥80% of cases.

---

## 4. Consequential Validity

### 4.1 The Core Question

Does using this framework actually change anything for affected communities, or is it just auditing theater?

### 4.2 Pilot Validation Design

**Setting:** One organization, one dataset, one community.

**Pre-audit:** Measure baseline outcome rates by group.

**Audit:** Run community-valid audit. Identify flagged groups. Apply reweighting recommendations to decision process.

**Post-audit (6-month follow-up):** Measure outcome rates by group again.

**Success condition:** Flagged groups show statistically significant improvement in outcome rates post-intervention, compared to non-intervention control (if available) or pre-audit baseline.

### 4.3 Minimum Viable Pilot

If a full RCT is not feasible, a case study with pre/post measurement is acceptable for publication as a proof of concept, with explicit acknowledgment of limitations.

---

## 5. Failure Modes and Safeguards

| Failure Mode | Description | Safeguard |
|---|---|---|
| Gaming | Organization manipulates dataset to pass audit | Audit requires raw decision data, not pre-processed summaries |
| Proxy discrimination | Protected attribute removed but proxies remain | Scope note: v1 audits observed outcomes only; proxy detection is future work |
| Small group instability | Groups with n<30 produce unreliable rates | Flag groups with n<30 as "insufficient sample" rather than computing DI |
| Community capture | Vocal minority defines configuration for whole community | Input protocol requires documented participant count and demographic representation |
| Config staleness | Community config set once, never updated | Config must include input_date; configs older than 24 months trigger re-elicitation warning |

---

## 6. Publication Validation Checklist

Before submitting to ACM FAccT or equivalent:

- [ ] Unit test coverage ≥ 90% on core functions
- [ ] EEOC consistency verified on ≥3 published case datasets
- [ ] At least one construct validity study (Phase 1–3 above) completed
- [ ] At least one pilot with pre/post outcome measurement documented
- [ ] Small group safeguard (n<30) implemented and tested
- [ ] Config staleness warning implemented
- [ ] Traceability record produced for every community-valid audit in the study
- [ ] Community representatives reviewed and approved use of their input in publication

---

*Framework documentation complete. Steps 1–5 done.*
