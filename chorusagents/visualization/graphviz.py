"""Graphviz renderer: exports agent networks as SVG, PNG, or PDF."""

from __future__ import annotations

import os
from typing import Optional

from chorusagents.core.architect import NetworkBlueprint
from chorusagents.core.topology import EdgeDirection, TopologyType

_TOPOLOGY_RANKDIR = {
    TopologyType.PIPELINE: "LR",
    TopologyType.HIERARCHICAL: "TB",
    TopologyType.STAR: "TB",
    TopologyType.MESH: "TB",
    TopologyType.CUSTOM: "TB",
}

_NODE_COLORS = [
    "#4A90D9", "#E67E22", "#2ECC71", "#9B59B6",
    "#E74C3C", "#1ABC9C", "#F39C12", "#34495E",
]


class GraphvizRenderer:
    """
    Renders an ChorusAgents NetworkBlueprint using the Graphviz library.

    Requires the optional ``visualization`` extra::

        pip install chorusagents[visualization]

    Or separately::

        pip install graphviz

    Example::

        renderer = GraphvizRenderer()
        renderer.render_to_file(blueprint, "network", fmt="svg")
        # Saves network.svg in the current directory
    """

    def render(self, blueprint: NetworkBlueprint) -> "graphviz.Digraph":  # type: ignore
        """Build and return a ``graphviz.Digraph`` object."""
        gv = self._import_graphviz()

        rankdir = _TOPOLOGY_RANKDIR.get(blueprint.topology_type, "TB")
        dot = gv.Digraph(
            name=blueprint.meta_role,
            graph_attr={
                "rankdir": rankdir,
                "bgcolor": "#FAFAFA",
                "fontname": "Helvetica",
                "label": f"{blueprint.meta_role}  [{blueprint.topology_type.value}]",
                "labelloc": "t",
                "fontsize": "16",
                "pad": "0.5",
                "splines": "ortho",
            },
            node_attr={
                "shape": "box",
                "style": "filled,rounded",
                "fontname": "Helvetica",
                "fontsize": "12",
                "margin": "0.2,0.1",
            },
            edge_attr={
                "fontname": "Helvetica",
                "fontsize": "10",
                "color": "#666666",
            },
        )

        # Add nodes
        agent_list = blueprint.agents
        for i, agent in enumerate(agent_list):
            color = _NODE_COLORS[i % len(_NODE_COLORS)]
            dot.node(
                agent.name,
                label=f"{agent.name}\n{agent.sub_role}",
                fillcolor=color,
                fontcolor="white",
            )

        # Add edges
        seen: set[tuple[str, str]] = set()
        for edge in blueprint.edges:
            if edge.direction == EdgeDirection.BIDIRECTIONAL:
                key = tuple(sorted([edge.source, edge.target]))
                if key not in seen:
                    dot.edge(
                        edge.source,
                        edge.target,
                        label=edge.label,
                        dir="both",
                        color="#333333",
                    )
                    seen.add(key)
            else:
                if (edge.source, edge.target) not in seen:
                    dot.edge(
                        edge.source,
                        edge.target,
                        label=edge.label,
                    )
                    seen.add((edge.source, edge.target))

        return dot

    def render_to_file(
        self,
        blueprint: NetworkBlueprint,
        output_path: str = "agent_network",
        fmt: str = "svg",
        view: bool = False,
    ) -> str:
        """
        Render and save the network diagram.

        Parameters
        ----------
        blueprint:
            The network blueprint to visualize.
        output_path:
            Output file path (without extension).
        fmt:
            Output format: ``"svg"``, ``"png"``, or ``"pdf"``.
        view:
            If True, open the rendered file automatically.

        Returns
        -------
        str
            Path to the saved file.
        """
        dot = self.render(blueprint)
        out = dot.render(output_path, format=fmt, view=view, cleanup=True)
        print(f"Network diagram saved to: {out}")
        return out

    def render_to_svg_string(self, blueprint: NetworkBlueprint) -> str:
        """Return the SVG diagram as a string."""
        dot = self.render(blueprint)
        return dot.pipe(format="svg").decode("utf-8")

    @staticmethod
    def _import_graphviz():
        try:
            import graphviz
            return graphviz
        except ImportError as e:
            raise ImportError(
                "The 'graphviz' package is required for GraphvizRenderer. "
                "Install it with: pip install chorusagents[visualization]"
            ) from e
