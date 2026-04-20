"""Tests for topology types and edge models."""

import pytest
from chorusagents.core.topology import TopologyType, TopologyEdge, EdgeDirection


def test_topology_type_values():
    assert TopologyType.STAR == "star"
    assert TopologyType.PIPELINE == "pipeline"
    assert TopologyType.MESH == "mesh"
    assert TopologyType.HIERARCHICAL == "hierarchical"
    assert TopologyType.CUSTOM == "custom"


def test_topology_edge_defaults():
    edge = TopologyEdge(source="A", target="B")
    assert edge.direction == EdgeDirection.BIDIRECTIONAL
    assert edge.label == ""
    assert edge.weight == 1.0


def test_topology_edge_unidirectional():
    edge = TopologyEdge(
        source="A", target="B",
        direction=EdgeDirection.UNIDIRECTIONAL,
        label="data flow"
    )
    assert edge.direction == EdgeDirection.UNIDIRECTIONAL
    assert edge.label == "data flow"


def test_topology_edge_repr():
    edge = TopologyEdge(source="X", target="Y", direction=EdgeDirection.BIDIRECTIONAL)
    assert "↔" in repr(edge)


def test_topology_edge_unidirectional_repr():
    edge = TopologyEdge(source="X", target="Y", direction=EdgeDirection.UNIDIRECTIONAL)
    assert "→" in repr(edge)
