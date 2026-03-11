"""
Pydantic request/response models for the fairness audit API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JSONAuditRequest(BaseModel):
    data: list[dict[str, Any]] = Field(
        ..., description="List of row dicts representing the dataset.", min_length=1
    )
    race_col: str = Field(..., description="Column name containing racial/group identifiers.", min_length=1)
    outcome_col: str = Field(..., description="Column name containing the outcome variable.", min_length=1)
    favorable_value: str = Field(
        ..., description="The outcome value considered favorable (e.g. 'hired', '1').", min_length=1
    )
    privileged_group: str | None = Field(
        default=None,
        description="Optional reference group for Disparate Impact calculation. "
        "Defaults to the group with the highest favorable outcome rate.",
    )


class JSONReweightRequest(BaseModel):
    data: list[dict[str, Any]] = Field(
        ..., description="List of row dicts representing the dataset.", min_length=1
    )
    race_col: str = Field(..., description="Column name containing racial/group identifiers.", min_length=1)
    outcome_col: str = Field(..., description="Column name containing the outcome variable.", min_length=1)
    favorable_value: str = Field(
        ..., description="The outcome value considered favorable.", min_length=1
    )
