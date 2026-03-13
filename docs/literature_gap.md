# Literature Gap Analysis
## Adaptive Racial Fairness Framework

**Version:** 0.1 (Draft)
**Status:** Internal — not for distribution

---

## 1. Overview

This document surveys the existing landscape of algorithmic fairness research and tooling, identifies what each approach contributes, and precisely locates the gap this framework fills. The purpose is to establish that the community-defined fairness concept is both novel and necessary — not duplicating existing work.

---

## 2. The Three Generations of Fairness Work

### Generation 1 — Metric Definition (2016–2019)
Early fairness research focused on formally defining what "fair" means mathematically. Key works:

- **Chouldechova (2017)** — proved that common fairness metrics are mathematically incompatible when base rates differ across groups. A system cannot simultaneously satisfy calibration and equal false positive rates if groups have different base rates. *Contribution: established the impossibility results that frame all subsequent debate. Gap: treats fairness as a purely mathematical question with no community input.*

- **Kleinberg et al. (2016)** — formalized the same incompatibility from a statistical perspective. *Same gap.*

- **Hardt, Price & Srebro (2016)** — introduced Equalized Odds and Equal Opportunity as post-processing fairness constraints. *Contribution: actionable mitigation. Gap: the constraint is defined by the researcher, not the community.*

**What Generation 1 established:** You cannot satisfy all fairness metrics simultaneously. You have to choose. *Who chooses is not addressed.*

---

### Generation 2 — Toolkits and Mitigation (2018–2022)
The field moved from theory to practice, building tools organizations could use.

- **IBM AIF360 (2018)** — comprehensive library of 70+ fairness metrics and mitigation algorithms. Industry standard. *Gap: metrics are selected by the auditor. Community plays no role in the configuration.*

- **Google What-If Tool (2018)** — interactive visualization for exploring model fairness. *Gap: exploratory only, no community input mechanism, no formal output.*

- **Microsoft Fairlearn (2020)** — mitigation library focused on group fairness constraints via ExponentiatedGradient and other algorithms. Used in this framework's `adversarial_fairlearn.py`. *Gap: fairness constraint is a technical parameter, not a community decision.*

- **LinkedIn's Fairness Toolkit** — internal, partially published. Focuses on representational fairness in feed ranking. *Gap: proprietary, not designed for external or community audits.*

**What Generation 2 established:** Mitigation is technically possible. Tools exist. *Who defines the target the mitigation aims for is still not addressed.*

---

### Generation 3 — Governance and Accountability (2021–present)
Research and policy shifted toward asking who is responsible for fairness and how accountability is enforced.

- **NYC Local Law 144 (2023)** — first U.S. law requiring automated employment decision tools (AEDTs) to undergo annual bias audits. Audits must be conducted by independent third parties and results published. *Contribution: legal mandate for auditing. Gap: the law specifies what must be measured but gives the auditor full discretion over the fairness standard. The affected community has no formal role.*

- **EU AI Act (2024)** — classifies certain AI systems as high-risk and requires conformity assessments including bias testing. *Gap: assessment criteria are regulatory, not community-defined.*

- **Algorithmic Justice League (Buolamwini et al.)** — advocacy and research organization focused on the social impact of AI. *Contribution: public legitimacy and awareness. Gap: advocacy, not a formal audit protocol.*

- **Data & Society / AI Now Institute** — critical research on power and AI. *Contribution: naming the structural issues. Gap: not building operational solutions.*

- **Participatory ML literature (Birhane, Kulynych, Sloane et al., 2022)** — emerging research strand arguing that affected communities should participate in ML system design. *Contribution: the closest existing work to this framework's core claim. Gap: participation is described as desirable but no operational protocol exists for how community input formally enters the fairness configuration of an audit system.*

**What Generation 3 established:** Accountability requires governance. Governance requires community voice. *How to operationalize community voice in a formal audit system has not been built.*

---

## 3. The Precise Gap

The table below maps what exists against what this framework contributes:

| Capability | AIF360 | Fairlearn | NYC LL144 | Participatory ML Literature | This Framework |
|---|---|---|---|---|---|
| Computes disparate impact | ✓ | ✓ | Required | — | ✓ |
| Offers mitigation algorithms | ✓ | ✓ | — | — | ✓ |
| Community defines fairness target | ✗ | ✗ | ✗ | Proposed, not built | **✓** |
| Machine-readable community config | ✗ | ✗ | ✗ | ✗ | **✓** |
| Traceable community input record | ✗ | ✗ | ✗ | ✗ | **✓** |
| Designed for non-technical auditors | Partial | ✗ | ✗ | — | **✓** |
| Pilot-ready for city government | ✗ | ✗ | Compliance only | ✗ | **✓** |

---

## 4. The Theoretical Anchor

The participatory ML literature (Birhane et al., 2022; Sloane et al., 2022) establishes that:

1. Affected communities have domain knowledge about fairness that researchers lack.
2. Participation improves both the legitimacy and accuracy of fairness assessments.
3. Current tools treat community input as optional consultation, not structural input.

This framework operationalizes point 3. It is the first system in which community-defined fairness parameters are a **required structural input** — the audit cannot run without a community configuration, and the configuration must be traceable to a defined input process.

---

## 5. What This Means for Publication

The publishable claim is narrow and strong:

> *We present the first operational framework in which fairness audit targets are formally specified by the affected community and machine-readable community configurations are a required structural input to the audit pipeline. We demonstrate this framework on [dataset/domain] and show that community-defined targets differ systematically from researcher-defined defaults in [X]% of cases, with implications for [outcome].*

This is citable, reproducible, and fills a documented gap in the ACM FAccT / NeurIPS fairness literature.

---

*Next: Algorithm Specification — Mathematical Formalization (Step 3)*
