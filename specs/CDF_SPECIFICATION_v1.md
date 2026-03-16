# Community-Defined Fairness (CDF) Specification v1.0

**Status:** Draft Standard
**Version:** 1.0.0
**Date:** 2026-03-15
**Schema:** `community_fairness_config_v1.schema.json`

---

## 1. Purpose

This specification defines a standard interchange format for **community-defined algorithmic fairness configurations**. A CDF configuration encodes what a specific community has decided constitutes "fair" for a specific algorithmic decision system.

The CDF format is designed to be:
- **Tool-agnostic:** Any fairness toolkit (AIF360, Fairlearn, custom systems) can consume it
- **Provenance-bearing:** Every configuration traces back to a documented community input process
- **Machine-readable:** JSON format with a formal JSON Schema
- **Domain-portable:** Works across employment, lending, criminal justice, healthcare, and education

---

## 2. Core Concepts

### 2.1 Community Fairness Configuration

A Community Fairness Configuration (CFC) is a JSON document that encodes three community decisions:

| Parameter | Type | Description |
|---|---|---|
| `priority_groups` | array[string] | Racial/ethnic groups the community identified for elevated scrutiny |
| `fairness_target` | string | The reference group whose outcome rate sets the benchmark |
| `fairness_threshold` | number (0,1] | The minimum acceptable disparate impact ratio |

These three parameters, combined, determine the outcome of any fairness audit. The CDF specification's contribution is formalizing that these parameters come from the community, not the auditor.

### 2.2 Audit Classification

Any audit produced using a CDF configuration carries one of three classifications:

| Classification | Meaning | Requirements |
|---|---|---|
| `community_valid` | Parameters set by a documented community process | Full provenance with ≥10 participants |
| `low_confidence` | Community input collected but below minimum threshold | Provenance present, <10 participants |
| `standard` | Parameters set by researcher, regulator, or auditor | No community provenance |

### 2.3 Provenance Record

A provenance record is a structured metadata block that links a configuration to its source. It answers: **who** defined these parameters, **when**, **where**, **how**, and with **how many people**.

Provenance is what distinguishes a CDF configuration from an arbitrary threshold choice. Without provenance, a configuration is classified as `standard` regardless of its parameter values.

---

## 3. Configuration Format

### 3.1 Required Fields

```json
{
  "cdf_version": "1.0",
  "priority_groups": ["Black or African American", "Native American"],
  "fairness_target": "White",
  "fairness_threshold": 0.9,
  "provenance": {
    "record_id": "550e8400-e29b-41d4-a716-446655440000",
    "input_protocol": "community_session",
    "input_date": "2026-04-01",
    "input_participants": 14
  }
}
```

### 3.2 Optional Fields

| Field | Type | Description |
|---|---|---|
| `domain` | string | Decision domain (employment, lending, criminal_justice, healthcare, education, housing, benefits, other) |
| `jurisdiction` | string | Geographic or institutional scope |
| `provenance.input_location` | string | Where the session took place |
| `provenance.facilitator` | string | Who facilitated |
| `provenance.facilitator_independent` | boolean | Whether facilitator is independent of the audited org |
| `provenance.demographic_representation` | object | Participant breakdown by group |
| `provenance.decision_method` | object | How each parameter was decided |
| `provenance.threshold_distribution` | object | Individual participant threshold responses |
| `provenance.dissent` | array | Recorded dissenting positions |
| `provenance.notes` | string | Session notes |
| `provenance.expires_at` | datetime | When re-elicitation is recommended |
| `audit_classification` | string | community_valid, low_confidence, or standard |
| `custom_metrics` | array[string] | Additional metrics requested by community |

### 3.3 Schema

The authoritative JSON Schema is at `community_fairness_config_v1.schema.json`. All CDF-compliant tools MUST validate configurations against this schema before use.

---

## 4. Conformance Requirements

### 4.1 For Configuration Producers

A tool that produces CDF configurations MUST:
1. Generate a UUID v4 for `provenance.record_id`
2. Record `input_date` as ISO 8601 date
3. Record `input_participants` as an integer ≥ 1
4. Set `cdf_version` to "1.0"
5. Validate the output against the CDF v1.0 JSON Schema

A tool that produces CDF configurations SHOULD:
1. Record demographic representation of participants
2. Record the decision method for each parameter
3. Record dissenting positions
4. Set `expires_at` to 24 months from `input_date`

### 4.2 For Configuration Consumers

A tool that consumes CDF configurations MUST:
1. Validate the configuration against the CDF v1.0 JSON Schema
2. Use `priority_groups`, `fairness_target`, and `fairness_threshold` from the configuration — not from auditor discretion
3. Include the `audit_classification` in all output
4. Include the `provenance.record_id` in all output
5. Clearly distinguish between `community_valid` and `standard` audits

A tool that consumes CDF configurations SHOULD:
1. Warn if the configuration has expired (past `expires_at`)
2. Warn if `input_participants` < 10
3. Include the full provenance record in audit output

### 4.3 For Configuration Registries

A registry that hosts CDF configurations MUST:
1. Validate all configurations against the CDF v1.0 JSON Schema before publishing
2. Organize configurations by jurisdiction and domain
3. Maintain a machine-readable index
4. Reject configurations without provenance records

---

## 5. Interoperability

### 5.1 AIF360

CDF configurations map to AIF360 as follows:
- `fairness_target` → `privileged_groups`
- `priority_groups` → `unprivileged_groups`
- `fairness_threshold` → comparison threshold for `BinaryLabelDatasetMetric.disparate_impact()`

### 5.2 Fairlearn

CDF configurations map to Fairlearn as follows:
- `fairness_target` → reference group in `MetricFrame`
- `fairness_threshold` → comparison threshold for `selection_rate` ratio
- `priority_groups` → groups of interest in disaggregated metrics

### 5.3 EEOC Compliance

When `fairness_threshold` = 0.80, the CDF audit is equivalent to the EEOC Uniform Guidelines four-fifths rule (1978). CDF-compliant audits at θ=0.80 produce results consistent with federal disparate impact standards.

---

## 6. Versioning

This specification follows semantic versioning. Version 1.0 is the initial release. Future versions may add:
- Support for non-racial protected attributes
- Continuous outcome metrics
- Multi-threshold configurations (different thresholds for different groups)
- Intersectional group definitions

Backward compatibility: any valid CDF v1.0 configuration MUST remain valid in future minor versions (1.x). Breaking changes require a major version increment.

---

## 7. Intellectual Property

This specification is published under the Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0). Anyone may implement, extend, or build upon this specification. Attribution is required.

Community configurations published using this format remain the intellectual property of the communities that produced them. Authorship credit belongs to the community, not the tool operator.

---

## 8. Reference Implementation

The reference implementation is the Adaptive Racial Fairness Framework:
- Configuration producer: `community_input.py` → `build_community_config()`
- Configuration consumer: `api/main.py` → `/audit/compliance` endpoint
- Integration adapters: `integrations/aif360_adapter.py`, `integrations/fairlearn_adapter.py`
- Reproducibility verification: `reproduce.py`
- Community input protocol: `docs/community_input_protocol.md`

Source code: [GitHub repository URL]
