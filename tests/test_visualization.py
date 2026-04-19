"""Tests for visualization renderers."""

import pytest

from agentfabric.visualization.mermaid import MermaidRenderer
from agentfabric.core.topology import TopologyType


def test_mermaid_render_produces_string(sample_blueprint):
    renderer = MermaidRenderer()
    diagram = renderer.render(sample_blueprint)
    assert isinstance(diagram, str)
    assert "Director" in diagram
    assert "Engineer" in diagram


def test_mermaid_render_contains_graph_keyword(sample_blueprint):
    renderer = MermaidRenderer()
    diagram = renderer.render(sample_blueprint)
    assert diagram.startswith("graph")


def test_mermaid_render_to_markdown(sample_blueprint):
    renderer = MermaidRenderer()
    md = renderer.render_to_markdown(sample_blueprint)
    assert md.startswith("```mermaid")
    assert md.endswith("```")


def test_mermaid_pipeline_uses_lr(sample_blueprint):
    sample_blueprint.topology_type = TopologyType.PIPELINE
    renderer = MermaidRenderer()
    diagram = renderer.render(sample_blueprint)
    assert "graph LR" in diagram


def test_mermaid_render_to_file(tmp_path, sample_blueprint):
    renderer = MermaidRenderer()
    out = tmp_path / "test.mmd"
    renderer.render_to_file(sample_blueprint, str(out))
    assert out.exists()
    content = out.read_text()
    assert "Director" in content


def test_mermaid_bidirectional_edge_uses_double_arrow(sample_blueprint):
    renderer = MermaidRenderer()
    diagram = renderer.render(sample_blueprint)
    assert "<-->" in diagram


def test_graphviz_import_error_on_missing_package(sample_blueprint, monkeypatch):
    """GraphvizRenderer raises ImportError when graphviz is not installed."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "graphviz":
            raise ImportError("No module named 'graphviz'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from agentfabric.visualization.graphviz import GraphvizRenderer
    renderer = GraphvizRenderer()
    with pytest.raises(ImportError, match="graphviz"):
        renderer.render(sample_blueprint)
