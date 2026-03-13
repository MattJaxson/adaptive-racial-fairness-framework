# System Architecture
## Adaptive Racial Fairness Framework

**Version:** 0.1 (Draft)
**Status:** Internal — not for distribution

---

## 1. Architecture Overview

The framework has three distinct layers. Each layer can operate independently or as part of the full stack.

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 3: GOVERNANCE                       │
│              Community Input → Configuration → Traceability      │
├─────────────────────────────────────────────────────────────────┤
│                         LAYER 2: AUDIT                           │
│              Data Ingestion → Metric Computation → Report        │
├─────────────────────────────────────────────────────────────────┤
│                       LAYER 1: MITIGATION                        │
│              Reweighting → Adversarial Debiasing → Validation    │
└─────────────────────────────────────────────────────────────────┘
```

Most organizations start at Layer 2. Layer 3 is what makes this framework community-valid and distinguishes it from all existing tools. Layer 1 is the remediation path after an audit surfaces disparities.

---

## 2. Layer 3 — Community Governance Layer

This is the novel layer. It does not exist in any other fairness framework.

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMMUNITY INPUT PROTOCOLS                      │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Voice Survey │  │ Web Survey   │  │ Community Session    │  │
│  │ (IVR/Phone)  │  │ (Form-based) │  │ (Facilitated)        │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └─────────────────┴──────────────────────┘              │
│                            │                                      │
│                            ▼                                      │
│              ┌─────────────────────────┐                         │
│              │  Input Processing Layer  │                         │
│              │  - Normalize responses   │                         │
│              │  - Resolve conflicts     │                         │
│              │  - Record provenance     │                         │
│              └────────────┬────────────┘                         │
│                           │                                       │
│                           ▼                                       │
│              ┌─────────────────────────┐                         │
│              │  community_definitions  │                         │
│              │       .json             │                         │
│              │                         │                         │
│              │  {                      │                         │
│              │   priority_groups,      │                         │
│              │   fairness_target,      │                         │
│              │   threshold,            │                         │
│              │   input_protocol,       │                         │
│              │   input_date,           │                         │
│              │   participants          │                         │
│              │  }                      │                         │
│              └────────────┬────────────┘                         │
│                           │                                       │
│                           ▼                                       │
│              ┌─────────────────────────┐                         │
│              │   Traceability Record   │                         │
│              │   (immutable log)       │                         │
│              └─────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

**Current state:** `community_definitions.json` exists. Input processing and traceability record are not yet built. The voice survey (Emily/District 3) is a candidate for the IVR input protocol.

**What needs to be built:** Input processing layer that translates community session responses into a valid config file, with provenance metadata.

---

## 3. Layer 2 — Audit Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                          AUDIT LAYER                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     DATA INGESTION                        │   │
│  │                                                            │   │
│  │   CSV Upload ──► Column Mapping ──► DataFrame             │   │
│  │   JSON POST  ──► Validation     ──► DataFrame             │   │
│  └─────────────────────────┬──────────────────────────────── ┘  │
│                             │                                     │
│                             ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  CONFIGURATION LOAD                       │   │
│  │                                                            │   │
│  │   community_definitions.json ──► C = (G, r, θ)           │   │
│  │   [Falls back to defaults if not present]                 │   │
│  └─────────────────────────┬──────────────────────────────── ┘  │
│                             │                                     │
│                             ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  METRIC COMPUTATION                       │   │
│  │                                                            │   │
│  │   group_outcomes_by_race()                                │   │
│  │   ├── P(Y=1 | A=a) for each a ∈ A                        │   │
│  │   ├── DI(a, r) = P(Y=1|A=a) / P(Y=1|A=r)                │   │
│  │   └── Flag: DI < θ                                        │   │
│  │                                                            │   │
│  │   calculate_racial_bias_score()                           │   │
│  │   └── DS = max_rate − min_rate                            │   │
│  └─────────────────────────┬──────────────────────────────── ┘  │
│                             │                                     │
│                             ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    AUDIT OUTPUT                           │   │
│  │                                                            │   │
│  │   JSON Report ──► API response / download                 │   │
│  │   CSV Report  ──► Row-level weights + flags               │   │
│  │   PDF Report  ──► [planned] Human-readable summary        │   │
│  │   Dashboard   ──► Dash app (local) / web (planned)        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Current state:** Fully built. FastAPI endpoints `/audit` and `/reweight` operational. Dash dashboard operational locally.

---

## 4. Layer 1 — Mitigation Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                        MITIGATION LAYER                          │
│                                                                   │
│   Audit Output (flagged groups)                                  │
│          │                                                        │
│          ▼                                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   REWEIGHTING                            │   │
│   │   fairness_reweight.py                                   │   │
│   │   ├── Compute target rate from reference group           │   │
│   │   ├── Assign wᵢ per individual                           │   │
│   │   └── Output: dataset + sample_weight column             │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                             │                                     │
│                             ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │             ADVERSARIAL DEBIASING (optional)             │   │
│   │   adversarial_fairlearn.py                               │   │
│   │   ├── ExponentiatedGradient (fairlearn)                  │   │
│   │   ├── Demographic Parity / Equalized Odds constraint     │   │
│   │   └── Output: debiased model + performance report        │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                             │                                     │
│                             ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   POST-MITIGATION AUDIT                  │   │
│   │   Re-run Layer 2 on reweighted/debiased output           │   │
│   │   Verify DI(a,r) ≥ θ for all flagged groups              │   │
│   │   Document delta: before vs. after                       │   │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Current state:** Reweighting operational. Adversarial debiasing built but not wired to API. Post-mitigation audit loop not yet implemented.

---

## 5. Data Flow — End to End

```
Community Session
      │
      ▼
community_definitions.json  ◄──── (Layer 3: community defines the standard)
      │
      ▼
Organization uploads CSV / sends JSON to /audit
      │
      ▼
Column mapping: sensitive_attr, outcome, favorable_value
      │
      ▼
Metric computation with community config
      │
      ├──► DI(a,r) for all groups
      ├──► Disparity Score
      └──► Flags for groups below θ
      │
      ▼
Audit Report (JSON / CSV / PDF)
      │
      ├── If flagged: → /reweight → post-mitigation audit
      └── If clean: → certified (future)
```

---

## 6. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RENDER CLOUD                             │
│                                                                   │
│  ┌─────────────────────────┐   ┌──────────────────────────────┐ │
│  │   fairness-audit-api    │   │   fairness-audit-landing     │ │
│  │   (Python/uvicorn)      │   │   (Static HTML)              │ │
│  │                         │   │                              │ │
│  │   api/main.py           │   │   landing/index.html         │ │
│  │   ├── /audit            │◄──│   Live demo widget           │ │
│  │   ├── /reweight         │   │                              │ │
│  │   └── /health           │   │                              │ │
│  └─────────────────────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        LOCAL (dev only)                          │
│   deploy_dash_app.py — Dash dashboard at localhost:8050          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. What Needs to Be Built (Gap Map)

| Component | Status | Priority |
|---|---|---|
| Community input processing layer | Not built | High — enables community-valid audits |
| Traceability record / provenance log | Not built | High — required for publication claim |
| PDF report export | Not built | Medium — professional output |
| Post-mitigation audit loop | Not built | Medium — completes the remediation path |
| Adversarial debiasing wired to API | Not built | Low — advanced use case |
| Certification infrastructure | Not designed | Future |

---

*Next: Validation Methodology (Step 5)*
