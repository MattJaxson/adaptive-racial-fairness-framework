# Patent Claims Structure
## Method and System for Community-Governed Algorithmic Fairness Auditing with Provenance-Tracked Configuration

**Filing Type:** Utility Patent (Method and System)
**Classification:** G06F 21/00 (Security); G06Q 10/06 (Administration; Management)
**Inventor:** Matt Jackson
**Priority Date:** [File provisional ASAP to establish priority]

---

## Abstract

A computer-implemented method and system for auditing algorithmic decision
systems for racial fairness, wherein the fairness standard applied to the
audit is derived from a community-defined configuration with cryptographic
provenance tracking, rather than from static regulatory defaults or
researcher-selected parameters. The system classifies audit outputs as
"community-valid" or "standard" based on the presence and validity of a
provenance-verified community fairness configuration, enabling traceable
accountability for who defined the fairness standard used in any given audit.

---

## Independent Claims

### Claim 1 — Method Claim (Core Pipeline)

A computer-implemented method for auditing an algorithmic decision system
for disparate impact across demographic groups, comprising:

**(a)** receiving, by a processor, a community fairness configuration (CFC)
comprising:
  - (i) one or more priority demographic groups identified by affected
    community members;
  - (ii) a reference demographic group against which outcome rates are
    compared;
  - (iii) a fairness threshold value (θ) between 0 and 1 representing the
    minimum acceptable disparate impact ratio;
  - (iv) a provenance record comprising a unique identifier, an input
    protocol type, an input date, a participant count, and a facilitator
    identifier;

**(b)** validating, by the processor, the community fairness configuration
by verifying the presence and completeness of the provenance record,
including verifying that the unique identifier is non-null, the input date
is parseable and within a staleness threshold, and the input protocol is
one of a set of recognized collection methods;

**(c)** classifying, by the processor, the audit as "community-valid" when
the community fairness configuration passes validation, and as "standard"
when it does not;

**(d)** receiving a dataset comprising demographic group labels and binary
outcome values for a plurality of individuals;

**(e)** computing, for each demographic group in the dataset, a disparate
impact ratio by dividing the favorable outcome rate of that group by the
favorable outcome rate of the reference group specified in the CFC;

**(f)** comparing each computed disparate impact ratio against the fairness
threshold (θ) specified in the CFC, and flagging groups whose ratio falls
below θ;

**(g)** generating an audit report that includes:
  - the audit classification (community-valid or standard),
  - the provenance record of the CFC used,
  - per-group outcome rates and disparate impact ratios,
  - the list of flagged groups,
  - the fairness threshold applied; and

**(h)** outputting the audit report in a machine-readable format that
preserves the link between the audit results and the specific community
fairness configuration that governed the audit.


### Claim 2 — System Claim

A system for community-governed algorithmic fairness auditing, comprising:

**(a)** a community configuration module configured to:
  - accept priority groups, a reference group, and a fairness threshold
    from a community input session;
  - generate a unique provenance record including a UUID, timestamp,
    participant count, input protocol type, and facilitator identifier;
  - persist the configuration as a structured document conforming to a
    defined schema (CDF v1.0);

**(b)** a configuration validation module configured to:
  - verify the completeness of the provenance record;
  - check configuration staleness against a configurable time threshold;
  - classify audits as "community-valid" or "standard" based on
    provenance validation results;

**(c)** an audit computation module configured to:
  - ingest tabular data with demographic group labels and binary outcomes;
  - compute disparate impact ratios using the reference group from the
    community configuration;
  - apply the community-defined fairness threshold to determine flagged
    groups;

**(d)** a report generation module configured to:
  - produce audit reports that embed the community configuration's
    provenance record;
  - clearly display the audit classification; and

**(e)** an interoperability layer configured to translate the community
fairness configuration into parameter formats accepted by third-party
fairness toolkits, including IBM AIF360 and Microsoft Fairlearn.


### Claim 3 — Interchange Format Claim

A computer-readable data structure for encoding a community-defined
fairness standard, comprising:

**(a)** a version identifier specifying the schema version;

**(b)** a priority groups array listing demographic groups designated for
elevated audit scrutiny;

**(c)** a fairness target string identifying the reference demographic group;

**(d)** a fairness threshold numeric value between 0 (exclusive) and 1
(inclusive);

**(e)** a provenance object comprising:
  - a universally unique identifier (UUID),
  - an input protocol enumeration value,
  - an ISO 8601 date string,
  - an integer participant count,
  - optional fields for facilitator name, decision method, threshold
    distribution statistics, and dissent record; and

**(f)** an audit classification enumeration indicating whether an audit
using this configuration qualifies as "community_valid," "standard," or
"low_confidence."

---

## Dependent Claims

### Claim 4
The method of Claim 1, wherein the provenance record further comprises
a dissent record capturing disagreements expressed during the community
input session, and a decision method field indicating whether the
configuration was reached by consensus, supermajority, or median vote.

### Claim 5
The method of Claim 1, further comprising a reweighting step wherein,
for each individual in a priority group, a sample weight is computed as:
  - w = target_rate / group_rate (for favorable outcomes), and
  - w = (1 - target_rate) / (1 - group_rate) (for unfavorable outcomes),
where target_rate is the favorable outcome rate of the reference group.

### Claim 6
The method of Claim 1, further comprising a compliance checking step
wherein the dataset is audited against a plurality of community fairness
configurations published in a registry, enabling comparative compliance
analysis across jurisdictions or community standards.

### Claim 7
The system of Claim 2, wherein the interoperability layer generates:
  - for IBM AIF360: a BinaryLabelDataset with privileged_groups and
    unprivileged_groups derived from the CFC; and
  - for Microsoft Fairlearn: constraint parameters for
    ExponentiatedGradient with DemographicParity threshold derived from
    the CFC fairness_threshold.

### Claim 8
The system of Claim 2, further comprising a configuration registry
module that stores published community fairness configurations indexed
by jurisdiction and domain, and provides a query interface for retrieving
applicable configurations.

### Claim 9
The method of Claim 1, wherein the staleness threshold for configuration
validation is configurable, and wherein configurations exceeding the
staleness threshold trigger a re-elicitation recommendation without
invalidating the audit classification.

### Claim 10
The data structure of Claim 3, further comprising a custom_metrics
array allowing communities to define domain-specific fairness metrics
beyond disparate impact ratio.

---

## Prosecution Strategy Notes

### What is patentable (novel + non-obvious):
1. **The governance-to-execution pipeline** — requiring a provenance-
   verified community config to classify an audit. No prior art combines
   community input provenance with audit classification.
2. **The CDF interchange format** — a standardized schema for encoding
   community fairness decisions with provenance. No equivalent exists.
3. **The dual classification system** — community_valid vs. standard
   based on provenance validation. Novel audit classification mechanism.

### What is NOT patentable (prior art):
- Disparate impact computation (EEOC 1978, well-known math)
- Sample reweighting for fairness (Kamiran & Calders 2012)
- ExponentiatedGradient / adversarial debiasing (Agarwal et al. 2018)
- The general concept of participatory design

### Recommended filing sequence:
1. **Provisional patent application** — file immediately to establish
   priority date. Cost: ~$300 (micro entity). Buys 12 months to file
   the full utility patent.
2. **Full utility patent** — file within 12 months of provisional.
   Focus claims on the governance pipeline (Claims 1-2) and the
   interchange format (Claim 3).
3. **PCT international application** — if pursuing international
   protection, file within 12 months of provisional.

### Key prior art to distinguish against:
- US Patent 10,885,528 (IBM) — "Fairness assessment of models"
  → IBM patent covers fairness *measurement*, not community governance
- US Patent 11,023,882 (Microsoft) — "Fairness in machine learning"
  → Microsoft patent covers mitigation algorithms, not provenance tracking
- Saleiro et al. 2018 (Aequitas) — fairness audit tool
  → No community input, no provenance, no configuration standard
