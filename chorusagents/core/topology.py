"""Topology types and edge definitions for agent networks."""

from enum import Enum
from typing import Literal
from pydantic import BaseModel


class TopologyType(str, Enum):
    """Communication topology patterns for agent networks."""

    STAR = "star"
    """One central hub agent communicates with all others (hub-and-spoke)."""

    PIPELINE = "pipeline"
    """Agents form a sequential chain; output of one feeds the next."""

    MESH = "mesh"
    """Every agent can communicate with every other agent (fully connected)."""

    HIERARCHICAL = "hierarchical"
    """Tree-like structure with supervisors and subordinates."""

    CUSTOM = "custom"
    """Arbitrary topology defined by explicit edge list."""


class EdgeDirection(str, Enum):
    BIDIRECTIONAL = "bidirectional"
    UNIDIRECTIONAL = "unidirectional"


class TopologyEdge(BaseModel):
    """A directed or bidirectional link between two agents."""

    source: str
    target: str
    direction: EdgeDirection = EdgeDirection.BIDIRECTIONAL
    label: str = ""
    weight: float = 1.0

    def __repr__(self) -> str:
        arrow = "↔" if self.direction == EdgeDirection.BIDIRECTIONAL else "→"
        return f"TopologyEdge({self.source} {arrow} {self.target})"
