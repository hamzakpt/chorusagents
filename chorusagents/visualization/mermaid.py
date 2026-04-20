"""Mermaid diagram renderer for agent networks."""

from __future__ import annotations

from chorusagents.core.architect import NetworkBlueprint
from chorusagents.core.topology import EdgeDirection, TopologyType


class MermaidRenderer:
    """
    Renders an ChorusAgents NetworkBlueprint as a Mermaid diagram string.

    The output can be embedded in Markdown, rendered in GitHub,
    or pasted into https://mermaid.live for interactive viewing.

    Example::

        renderer = MermaidRenderer()
        diagram = renderer.render(blueprint)
        print(diagram)
    """

    def render(self, blueprint: NetworkBlueprint) -> str:
        """Return a Mermaid graph definition string."""
        lines = [self._graph_type(blueprint), ""]

        # Subgraph for metadata
        lines.append(f'    %% Network: {blueprint.meta_role}')
        lines.append(f'    %% Topology: {blueprint.topology_type.value}')
        lines.append("")

        # Node definitions with labels
        for agent in blueprint.agents:
            safe_name = _safe_id(agent.name)
            lines.append(f'    {safe_name}["{agent.name}<br/><i>{agent.sub_role}</i>"]')

        lines.append("")

        # Edges
        seen: set[tuple[str, str]] = set()
        for edge in blueprint.edges:
            src = _safe_id(edge.source)
            tgt = _safe_id(edge.target)
            label = f'|"{edge.label}"|' if edge.label else ""

            if edge.direction == EdgeDirection.BIDIRECTIONAL:
                key = tuple(sorted([src, tgt]))
                if key not in seen:
                    lines.append(f"    {src} <--> {tgt}")
                    seen.add(key)
            else:
                if (src, tgt) not in seen:
                    lines.append(f"    {src} --{label}--> {tgt}")
                    seen.add((src, tgt))

        return "\n".join(lines)

    def render_to_file(self, blueprint: NetworkBlueprint, path: str) -> None:
        """Write the Mermaid diagram to a .mmd file."""
        diagram = self.render(blueprint)
        with open(path, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"Mermaid diagram saved to: {path}")

    def render_to_markdown(self, blueprint: NetworkBlueprint) -> str:
        """Wrap the diagram in a Markdown code fence."""
        return f"```mermaid\n{self.render(blueprint)}\n```"

    @staticmethod
    def _graph_type(blueprint: NetworkBlueprint) -> str:
        if blueprint.topology_type == TopologyType.PIPELINE:
            return "graph LR"
        return "graph TD"


def _safe_id(name: str) -> str:
    """Convert an agent name to a Mermaid-safe node ID."""
    return name.replace(" ", "_").replace("-", "_")
