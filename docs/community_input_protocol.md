# Community Input Protocol
## A Facilitator's Guide to Running a Community-Defined Fairness Session

**Version:** 1.0
**Status:** Ready for field use
**Time required:** 90 minutes
**Minimum participants:** 10 (recommended: 12–20)

---

## 1. What This Is

This protocol produces a **community fairness configuration** — a machine-readable file that encodes what a specific community believes "fair" means for a specific decision system. The configuration directly controls how the Adaptive Racial Fairness Framework audits that system.

This is not a survey. It is a structured decision-making session in which community members set three parameters that determine whether an algorithm passes or fails a fairness audit:

1. **Priority groups** — which racial/ethnic groups should receive elevated scrutiny
2. **Reference group** — which group's outcome rate is the benchmark
3. **Fairness threshold** — how close other groups must be to the reference group to be considered "fair"

---

## 2. Who Should Be in the Room

### Required representation
- At least 10 participants from the community affected by the decision system being audited
- Demographic composition should roughly reflect the community's racial/ethnic makeup
- At least 2 participants from each racial/ethnic group present in the dataset being audited
- At least 1 person with lived experience of the decision system (e.g., someone who has applied for a loan, been through the hiring process, or been subject to a risk assessment)

### Who should NOT be in the room
- Employees or representatives of the organization being audited (they may observe but not participate in parameter-setting)
- The auditor (they facilitate but do not vote)

### Facilitator requirements
- Independent of the organization being audited
- Trained in community facilitation (not just technically competent)
- Must not express opinions on what the parameters "should" be
- Records all decisions and dissenting views

---

## 3. Pre-Session Preparation

### Materials to prepare (facilitator)

1. **Dataset summary sheet** (1 page) showing:
   - What decision system is being audited (e.g., "Bank X's mortgage approval algorithm")
   - What the data contains: number of records, what the outcome is (approved/denied), what racial groups are present
   - Current outcome rates by group (a simple bar chart)
   - NO DI ratios or "flagged" labels — the community sets those, not the facilitator

2. **Plain-language threshold explainer** (1 page):
   - "The threshold is the minimum ratio between any group's approval rate and the reference group's approval rate."
   - Visual example: "If White applicants are approved 50% of the time and Black applicants are approved 40% of the time, the ratio is 0.80 (40/50). If the threshold is 0.80, this passes. If the threshold is 0.90, this fails."
   - Include 3 concrete scenarios at 0.7, 0.8, and 0.9 showing what passes/fails

3. **Decision cards** — physical or digital cards for each participant to record their individual choices before group discussion (prevents anchoring)

4. **Dissent recording form** — a structured way for participants to register disagreement with the final consensus

### Room setup
- Circle or U-shape seating (no "head of table")
- All materials in the primary language(s) of participants
- Childcare and food provided if possible
- Compensation for participants' time (recommended: $50–$100 gift card or equivalent)

---

## 4. Session Agenda (90 minutes)

### Opening (10 min)
- Facilitator introduces themselves and their independence from the organization being audited
- Explain: "You are here to set the fairness standard for [decision system]. Your decisions will directly control how this system is audited. There are no right or wrong answers — this is about what YOU believe is fair."
- Ground rules: every voice matters, no technical jargon, dissent is recorded and respected

### Part 1 — Understanding the Data (15 min)
- Present the dataset summary sheet
- Walk through the bar chart of outcome rates by group
- Answer questions — but do NOT interpret the data (e.g., do not say "this group is disadvantaged")
- Check: "Does everyone understand what the numbers represent?"

### Part 2 — Priority Groups (20 min)
**Question: "Which groups should we watch most carefully in this audit?"**

- Each participant writes their answer on a decision card (2 minutes, silent)
- Go around the circle: each person shares their choice and why
- Facilitator records all responses on a visible board
- Discussion: are there groups not in the data that should be? Are there groups that don't need elevated scrutiny?
- **Decision method:** A group is included as a priority group if ≥50% of participants name it
- Record the final list AND any dissenting views

### Part 3 — Reference Group (15 min)
**Question: "Which group's outcome rate should be the standard that other groups are compared against?"**

- Explain: "This is the group that sets the bar. If we compare everyone to this group, which group represents the outcome rate that everyone should have access to?"
- Each participant writes their answer on a decision card
- Share and discuss
- **Decision method:** Simple majority vote. If no majority, the group with the highest outcome rate in the data is used (explain this default)
- Record the decision

### Part 4 — Fairness Threshold (20 min)
**Question: "How close does every group need to be to the reference group for the system to be considered fair?"**

This is the hardest part. Use the concrete scenarios prepared earlier.

- Present three scenarios:
  - **0.70 (lenient):** "A group can be approved at 70% the rate of the reference group and still pass. For lending: if White applicants get 50% approval, Black applicants at 35% would pass."
  - **0.80 (current federal standard):** "A group can be approved at 80% the rate. Black applicants at 40% would pass."
  - **0.90 (strict):** "A group must be approved at 90% the rate. Black applicants at 40% would FAIL. They'd need 45% to pass."
- Each participant writes their preferred threshold on a decision card
- Share and discuss
- **Decision method:** Median of all individual responses (not mean — resistant to outliers)
- Record the decision AND the full distribution of individual responses

### Closing (10 min)
- Read back all three decisions
- Ask: "Does this reflect what this group believes is fair for this system?"
- Allow final objections — record them
- Explain what happens next: "These parameters will be encoded into the audit system. Every future audit of this system will use YOUR definition of fair. The audit report will say 'community-defined' and include the date, location, and number of participants in this session."
- Thank participants

---

## 5. Post-Session — Creating the Configuration

The facilitator enters the session results into the framework using `build_community_config()`:

```python
from community_input import build_community_config

config = build_community_config(
    priority_groups=["Black or African American", "Native American"],
    fairness_target="White",
    fairness_threshold=0.9,
    input_protocol="community_session",
    input_location="Detroit, MI — Cass Corridor Commons",
    input_participants=14,
    facilitator="[Facilitator Name]",
    notes="Consensus on priority groups. Threshold median: 0.9 (range: 0.8–0.95). "
          "Two participants advocated for 0.95; recorded as dissent. "
          "One participant noted Native American representation was only 1 person in room.",
    output_path="data/community_definitions.json",
)
```

This produces a JSON configuration with:
- A unique UUID (`record_id`)
- ISO 8601 timestamp
- All session metadata
- `audit_type: "community_valid"`

Every subsequent audit using this configuration will be classified as **community-valid** in the output, distinguishing it from researcher-default audits.

---

## 6. Validity Requirements

A community input session produces a valid configuration if ALL of the following are met:

| Requirement | Minimum | Recommended |
|---|---|---|
| Participants | 10 | 15–20 |
| Racial groups represented | ≥2 | All groups in dataset |
| Session length | 60 min | 90 min |
| Decision method documented | Yes | Yes |
| Dissent recorded | Yes | Yes |
| Facilitator independent | Yes | Yes |
| Participant compensation | Documented | Provided |
| Language accessibility | Primary language | Multiple languages |

Sessions with fewer than 10 participants are flagged as `low_confidence` in the provenance record. They are still valid but carry a warning.

---

## 7. Re-elicitation

Community configurations expire. The framework enforces a **24-month staleness window** — after 24 months, the system warns that the configuration should be refreshed through a new session.

Re-elicitation is also triggered if:
- The dataset changes substantially (new groups appear, outcome definitions change)
- The decision system being audited is materially altered
- Community representatives request it

---

## 8. Ethical Considerations

1. **Informed consent:** Participants must understand that their collective decisions will be encoded into a system and used to evaluate an organization. Consent must be documented.
2. **Anonymity:** Individual decision cards are recorded in aggregate. No individual participant's name is attached to a specific parameter choice.
3. **Power dynamics:** The facilitator must actively ensure that louder or more educated voices do not dominate. The decision card method (write first, then discuss) mitigates anchoring bias.
4. **Representation vs. delegation:** This session does not claim to represent "all Black people" or "all residents of Detroit." It represents the specific group of people who participated, and the configuration's provenance record makes this explicit.
5. **Right to withdraw:** Any participant can withdraw at any time. Their individual responses are removed from the aggregate.

---

## 9. Adaptations

### For city government use
- Embed the session in an existing public comment or town hall process
- City council can formally adopt the resulting configuration via resolution
- Configuration becomes part of the city's algorithmic accountability policy

### For organizational (corporate) use
- Employee resource groups or external community advisory boards serve as participants
- The organization commits in writing to use the resulting configuration for its next audit cycle
- Results are included in the published audit report (per NYC LL144 or equivalent)

### For academic research use
- Session requires IRB approval if results will be published
- Participants sign informed consent
- Full protocol and all aggregate decision data are included in the paper's supplementary materials

---

*This protocol is designed to be used with the Adaptive Racial Fairness Framework's `community_input.py` module. The output directly integrates with the `/audit`, `/audit/pdf`, and `/audit/remediate` API endpoints.*
