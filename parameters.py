"""Traceable numerical parameters.

Every quantity used by the model is classified as one of:
MEASURED, LITERATURE, STANDARD ATMOSPHERE, DERIVED, ASSUMED, USER INPUT.

Assumed values are never presented as experimental facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class Status(str, Enum):
    MEASURED = "MEASURED"
    LITERATURE = "LITERATURE"
    STANDARD_ATMOSPHERE = "STANDARD ATMOSPHERE"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    USER_INPUT = "USER INPUT"
    LITERATURE_BENCHMARK = "LITERATURE BENCHMARK"


@dataclass
class Parameter:
    name: str
    value: Any
    unit: str
    source: str
    status: Status
    uncertainty: float | None = None
    uncertainty_unit: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "status": self.status.value,
            "uncertainty": self.uncertainty,
            "uncertainty_unit": self.uncertainty_unit,
            "notes": self.notes,
        }


@dataclass
class ParameterLedger:
    """Collects every numerical input/output used in a run."""

    items: list[Parameter] = field(default_factory=list)

    def add(self, parameter: Parameter) -> Parameter:
        self.items.append(parameter)
        return parameter

    def record(
        self,
        name: str,
        value: Any,
        unit: str,
        source: str,
        status: Status,
        uncertainty: float | None = None,
        uncertainty_unit: str | None = None,
        notes: str = "",
    ) -> Any:
        self.add(
            Parameter(
                name=name,
                value=value,
                unit=unit,
                source=source,
                status=status,
                uncertainty=uncertainty,
                uncertainty_unit=uncertainty_unit,
                notes=notes,
            )
        )
        return value

    def to_rows(self) -> list[dict[str, Any]]:
        return [p.as_dict() for p in self.items]

    def extend(self, parameters: Iterable[Parameter]) -> None:
        self.items.extend(parameters)
