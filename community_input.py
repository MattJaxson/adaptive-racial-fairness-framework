"""
Community Input Processing Layer
---------------------------------
Converts raw community input (survey responses, session notes) into a
validated community_definitions.json config, with a provenance record
attached for traceability.

Usage:
    from community_input import build_community_config

    config = build_community_config(
        priority_groups=["Black", "Latinx"],
        fairness_target="White",
        fairness_threshold=0.8,
        input_protocol="voice_survey",
        input_location="Detroit District 3",
        input_participants=142,
        facilitator="Jane Smith",
        notes="Session held 2026-03-13. Consensus reached on priority groups.",
        output_path="data/community_definitions.json",
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VALID_PROTOCOLS = {"voice_survey", "web_survey", "community_session", "structured_interview", "other"}
MIN_PARTICIPANTS = 10  # Below this, config is flagged as low-confidence
STALE_MONTHS = 24      # Configs older than this trigger a re-elicitation warning


def build_community_config(
    priority_groups: list[str],
    fairness_target: str,
    fairness_threshold: float = 0.8,
    input_protocol: str = "community_session",
    input_location: str = "",
    input_participants: int = 0,
    facilitator: str = "",
    notes: str = "",
    output_path: Optional[str] = None,
) -> dict:
    """
    Build and optionally save a community configuration with provenance metadata.

    Parameters
    ----------
    priority_groups : list[str]
        Racial/ethnic groups the community identified as requiring elevated scrutiny.
    fairness_target : str
        The reference group whose outcome rate sets the fairness benchmark.
    fairness_threshold : float
        Disparate impact threshold below which a group is flagged (default: 0.8).
    input_protocol : str
        How input was collected. One of: voice_survey, web_survey,
        community_session, structured_interview, other.
    input_location : str
        Where the input session took place (city, district, organization).
    input_participants : int
        Number of community members who participated in the input session.
    facilitator : str
        Name or role of the person who facilitated the session.
    notes : str
        Free-text notes about the session or any unresolved disagreements.
    output_path : str, optional
        If provided, writes the config to this path as JSON.

    Returns
    -------
    dict
        The complete community configuration including provenance record.
    """
    # Validate
    if not priority_groups:
        raise ValueError("priority_groups cannot be empty.")
    if not fairness_target:
        raise ValueError("fairness_target cannot be empty.")
    if not (0.0 < fairness_threshold <= 1.0):
        raise ValueError("fairness_threshold must be between 0 and 1.")
    if input_protocol not in VALID_PROTOCOLS:
        raise ValueError(f"input_protocol must be one of: {VALID_PROTOCOLS}")

    low_confidence = input_participants > 0 and input_participants < MIN_PARTICIPANTS

    config = {
        # Core fairness parameters — these drive the audit
        "priority_groups": [str(g).strip() for g in priority_groups],
        "fairness_target": str(fairness_target).strip(),
        "fairness_threshold": fairness_threshold,

        # Provenance — required for community-valid audit classification
        "provenance": {
            "record_id": str(uuid.uuid4()),
            "input_protocol": input_protocol,
            "input_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "input_location": input_location,
            "input_participants": input_participants,
            "facilitator": facilitator,
            "notes": notes,
            "low_confidence": low_confidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },

        # Metadata
        "audit_type": "community_valid",
        "fairness_definition": "Community-driven definition of equity",
        "custom_metrics": [],
    }

    if low_confidence:
        logger.warning(
            "Community config created with only %d participants (minimum recommended: %d). "
            "Flagged as low_confidence.",
            input_participants,
            MIN_PARTICIPANTS,
        )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Community config written to %s (record_id: %s)", path, config["provenance"]["record_id"])

    return config


def validate_community_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate a community config dict. Returns (is_valid, list_of_issues).

    Also checks for staleness — configs older than STALE_MONTHS months
    are flagged for re-elicitation.
    """
    issues = []

    # Required fields
    if not config.get("priority_groups"):
        issues.append("Missing or empty priority_groups.")
    if not config.get("fairness_target"):
        issues.append("Missing fairness_target.")
    threshold = config.get("fairness_threshold")
    if threshold is None or not (0.0 < threshold <= 1.0):
        issues.append("fairness_threshold must be between 0 and 1.")

    # Provenance check
    provenance = config.get("provenance")
    if not provenance:
        issues.append(
            "No provenance record found. This config cannot be used for a community-valid audit."
        )
    else:
        if not provenance.get("record_id"):
            issues.append("Provenance record is missing record_id.")
        if not provenance.get("input_date"):
            issues.append("Provenance record is missing input_date.")
        if not provenance.get("input_protocol"):
            issues.append("Provenance record is missing input_protocol.")

        # Staleness check
        input_date_str = provenance.get("input_date")
        if input_date_str:
            try:
                input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
                months_old = (datetime.now() - input_date).days / 30
                if months_old > STALE_MONTHS:
                    issues.append(
                        f"Community config is {int(months_old)} months old "
                        f"(threshold: {STALE_MONTHS} months). Re-elicitation recommended."
                    )
            except ValueError:
                issues.append(f"Could not parse input_date: {input_date_str}")

    is_valid = len(issues) == 0
    return is_valid, issues


def is_community_valid(config: dict) -> bool:
    """
    Returns True if the config qualifies for a community-valid audit.
    A config is community-valid if it has a provenance record with no critical issues
    (staleness warnings do not disqualify it).
    """
    _, issues = validate_community_config(config)
    # Filter out staleness warnings — they're warnings, not disqualifiers
    critical_issues = [i for i in issues if "months old" not in i]
    return len(critical_issues) == 0
