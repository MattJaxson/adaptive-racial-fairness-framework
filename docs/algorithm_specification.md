# Algorithm Specification
## Adaptive Racial Fairness Framework — Mathematical Formalization

**Version:** 0.1 (Draft)
**Status:** Internal — not for distribution

---

## 1. Notation

| Symbol | Definition |
|---|---|
| D | Dataset of n individuals: D = {(xᵢ, aᵢ, yᵢ)} for i = 1…n |
| xᵢ | Feature vector for individual i |
| aᵢ ∈ A | Sensitive attribute (racial/ethnic group) for individual i |
| A | Set of all racial/ethnic groups present in D |
| yᵢ ∈ {0, 1} | Observed outcome for individual i (1 = favorable) |
| G ⊆ A | Priority groups — defined by the community configuration |
| r ∈ A | Reference group — defined by the community configuration |
| C | Community configuration: C = (G, r, θ) |
| θ | Community-defined fairness threshold (default: 0.8) |
| wᵢ | Sample weight assigned to individual i after reweighting |

---

## 2. Core Metrics

### 2.1 Group Outcome Rate

For each group a ∈ A, the favorable outcome rate is:

```
P(Y=1 | A=a) = Σ yᵢ · 𝟙[aᵢ = a] / Σ 𝟙[aᵢ = a]
```

Where 𝟙[·] is the indicator function.

In plain terms: the fraction of individuals in group a who received a favorable outcome.

---

### 2.2 Reference Group Selection

The reference group r is determined as follows:

```
If r is specified in community configuration C:
    Use r as specified.
    If r is not present in D:
        Fall back to: r* = argmax_{a ∈ A} P(Y=1 | A=a)

If r is not specified in C:
    r* = argmax_{a ∈ A} P(Y=1 | A=a)
```

Default behavior: reference group = "White" if present in dataset, otherwise the group with the highest outcome rate.

---

### 2.3 Disparate Impact Ratio

For each group a ∈ A, the Disparate Impact ratio with respect to reference group r is:

```
DI(a, r) = P(Y=1 | A=a) / P(Y=1 | A=r)
```

**Flag condition:** DI(a, r) < θ, where θ is the community-defined fairness threshold.
Default θ = 0.8 (EEOC 4/5ths rule).

A community may set θ higher (e.g., 0.9) to require a stricter standard.

---

### 2.4 Statistical Parity Gap

The maximum outcome disparity across all groups:

```
SPG(D) = max_{a ∈ A} P(Y=1 | A=a) − min_{a ∈ A} P(Y=1 | A=a)
```

Range: [0, 1]. Zero = perfect parity. One = maximum disparity.

---

### 2.5 Disparity Score

A normalized scalar summarizing total racial outcome inequality across the dataset:

```
DS(D) = max_{a ∈ A} P(Y=1 | A=a) − min_{a ∈ A} P(Y=1 | A=a)
```

Note: DS is equivalent to SPG in v1. Future versions will weight this by group size and community-defined priority weights.

---

## 3. Community-Defined Fairness (The Novel Contribution)

### 3.1 Community Configuration

The community configuration C is a structured input:

```json
{
  "priority_groups": ["Black", "Latinx"],
  "fairness_target": "White",
  "fairness_threshold": 0.8,
  "input_protocol": "voice_survey",
  "input_date": "2026-03-13",
  "input_location": "Detroit District 3",
  "input_participants": 142
}
```

**Critical distinction:** In all existing frameworks, these parameters are set by the auditor or researcher. In this framework, they are set by the community and the auditor is required to use them.

### 3.2 Formal Requirement

**Definition (Community-Valid Audit):** An audit of dataset D is community-valid if and only if:

1. A community configuration C has been produced through a documented input protocol.
2. The priority groups G and reference group r used in all metric calculations are drawn from C, not from auditor discretion.
3. The fairness threshold θ used for flagging is drawn from C.
4. C is stored with the audit output and traceable to its source.

An audit that uses researcher-selected parameters is a **standard audit**, not a community-valid audit. This framework supports both but distinguishes between them in all outputs.

---

## 4. Reweighting Algorithm

When a dataset fails community-defined fairness thresholds, the reweighting procedure adjusts sample weights to align priority group outcome rates toward the reference group rate.

### 4.1 Target Rate

For each priority group g ∈ G:

```
target_rate(g) = P(Y=1 | A=r)   [the reference group's outcome rate]
```

### 4.2 Weight Computation

For each individual i with aᵢ = g ∈ G:

```
If yᵢ = 1:
    wᵢ = target_rate(g) / P(Y=1 | A=g)

If yᵢ = 0:
    wᵢ = (1 − target_rate(g)) / (1 − P(Y=1 | A=g))
```

For individuals not in a priority group:

```
wᵢ = 1.0
```

### 4.3 Interpretation

Reweighting does not change the dataset — it assigns importance weights that a downstream model or analyst can use to train or evaluate systems that would produce fairer outcomes if the reweighted distribution were the true distribution.

---

## 5. Audit Output Specification

A complete audit of dataset D with community configuration C produces:

```
AuditResult = {
  dataset_summary: {
    n_records: int,
    n_groups: int,
    groups: [str],
    outcome_column: str,
    sensitive_column: str
  },
  community_config: C,
  group_metrics: {
    for each a ∈ A: {
      outcome_rate: P(Y=1 | A=a),
      disparate_impact: DI(a, r),
      flagged: bool  [DI(a,r) < θ]
    }
  },
  summary_metrics: {
    disparity_score: DS(D),
    statistical_parity_gap: SPG(D),
    n_groups_flagged: int
  },
  audit_type: "community_valid" | "standard",
  audit_timestamp: ISO8601
}
```

---

## 6. Complexity and Performance

- Time complexity: O(n) for metric computation over n records
- Space complexity: O(|A|) for group-level aggregates
- No model training required — purely statistical over observed outcomes
- Suitable for datasets from n=50 to n=10,000,000+

---

*Next: System Architecture Wireframe (Step 4)*
