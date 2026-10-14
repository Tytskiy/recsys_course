from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GroupDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Name of the group in the experiment"
    )
    range_start: int = Field(
        ge=0, description="Start of range, which contains all hashed IDs of the group"
    )
    range_stop: int = Field(
        ge=0, description="End of range, which contains all hashed IDs of the group"
    )
    overrides: dict[str, Any] = Field(
        description="Overrides of default parameter values for the group"
    )

    @model_validator(mode="after")
    def check_range_order(self) -> GroupDefinition:
        if self.range_stop < self.range_start:
            raise ValueError("range_stop must be >= range_start")
        return self


class ExperimentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    salt: str = Field(
        description="Salt for hashing IDs during splitting into groups"
    )
    mod: int = Field(
        gt=0, description="Modulo for hash of ID during splitting"
    )
    default: dict[str, Any] = Field(
        description="Parameter default values for the experiment"
    )
    groups: list[GroupDefinition] = Field(
        description="List of experiment groups"
    )


class ResolveResponse(BaseModel):
    group_name: str = Field(
        description="Group name, which contains the user"
    )
    params: dict[str, Any] = Field(
        description="Parameter set (default with overrides) of the group"
    )


class UpsertExperimentResponse(BaseModel):
    experiment_name: str = Field(description="Created experiment name")
