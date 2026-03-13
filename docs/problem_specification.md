# Problem Specification
## Adaptive Racial Fairness Framework

**Version:** 0.1 (Draft)
**Status:** Internal — not for distribution
**Authors:** [to be completed with co-author]

---

## 1. Problem Statement

Automated decision systems — used in hiring, lending, benefits distribution, housing, and criminal justice — produce outcomes that systematically differ across racial groups. These disparities are measurable, persistent, and often invisible to the organizations deploying the systems.

Current approaches to detecting and correcting these disparities share a critical structural flaw: **the definition of "fairness" is determined by researchers, regulators, or vendors — not by the communities most affected by the decisions.**

This produces three compounding failures:

1. **Measurement mismatch** — fairness metrics optimized for academic legibility do not necessarily reflect what affected communities experience as fair.
2. **Adoption failure** — frameworks imposed on communities without their input lack legitimacy and are resisted or ignored.
3. **Accountability gap** — without community-defined targets, there is no clear standard against which an organization can be held accountable.

---

## 2. Core Claim

**A fairness audit is only valid if the fairness standard was defined by the community subject to the decision system.**

This is the central, falsifiable claim of this framework. It distinguishes this work from all existing fairness toolkits, which treat the fairness definition as a technical parameter set by the auditor.

---

## 3. Scope

**In scope:**
- Binary and categorical outcome decisions (hired/not hired, approved/denied, flagged/cleared)
- Any decision system where a racial or ethnic group can be identified as a sensitive attribute
- Organizational audits (HR, lending, benefits, city services)
- Community-level governance of recurring algorithmic decisions

**Out of scope (v1):**
- Continuous outcome prediction (e.g., credit scoring as a number rather than approve/deny)
- Real-time intervention in live systems
- Non-racial protected attributes (gender, age, disability) — extensible in future versions
- Systems where ground truth outcomes are unavailable

---

## 4. The Gap This Fills

| Existing Framework | What It Does | What It Misses |
|---|---|---|
| IBM AIF360 | Broad fairness metric library | Fairness target is researcher-defined |
| Google What-If Tool | Visual exploration of model behavior | No community input mechanism |
| Microsoft Fairlearn | Mitigation algorithms | Defines fairness mathematically, not socially |
| NYC Local Law 144 | Mandates annual bias audits | Auditor defines the standard, not the community |
| Academic FAccT literature | Rigorous metric debate | Operates entirely within research institutions |

**None of the above give the affected community a formal role in defining the fairness target.**

This framework is the first to formalize community input as a required input to the audit — not an optional consultation, but a structural dependency.

---

## 5. Key Definitions

**Community-Defined Fairness Target:** A fairness threshold or reference condition specified by representatives of the group most affected by a decision system, captured through a defined input protocol (e.g., structured survey, community session, voice response system) and encoded as a machine-readable configuration.

**Disparate Impact:** The ratio of favorable outcome rates between a protected group and a reference group. A ratio below 0.8 triggers the 4/5ths rule threshold established by the EEOC (1978).

**Disparity Score:** A scalar summary of total outcome inequality across all racial groups in a dataset, defined as the difference between the maximum and minimum group-level favorable outcome rates.

**Priority Group:** A racial/ethnic group designated by the community as requiring elevated fairness scrutiny — typically groups with documented historical disadvantage in the decision domain.

**Reference Group:** The group whose outcome rate serves as the baseline for disparate impact calculations. Default: the group with the highest favorable outcome rate, or "White" if present in the dataset.

---

## 6. Success Conditions

This framework succeeds if:

1. An organization can run a complete audit using only a CSV of their decision data and a community-defined configuration file.
2. The audit output is interpretable by a non-technical city council member.
3. The community-defined fairness target is traceable — there is a documented record of who defined it, when, and how.
4. The framework produces results consistent with EEOC disparate impact standards where applicable.
5. A community can point to the fairness target and say: "We set that. It reflects what we believe is fair."

---

## 7. What This Is Not

- This is not a claim that disparity scores capture all dimensions of racial injustice.
- This is not a replacement for structural policy change.
- This is not a certification system (v1). Certification infrastructure is a future layer.
- This is not neutral. It is explicitly designed to center the perspective of communities that have historically been subject to, not participants in, algorithmic decision-making.

---

*Next: Literature Gap Document (Step 2)*
