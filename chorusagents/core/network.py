"""AgentNetwork: the runtime that routes queries through the synthesized agent graph."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Optional

import networkx as nx

from chorusagents.core.agent import Agent, AgentMessage, AgentResponse
from chorusagents.core.architect import NetworkBlueprint
from chorusagents.core.topology import TopologyType, TopologyEdge, EdgeDirection
from chorusagents.utils.logger import get_logger

logger = get_logger(__name__)

# Callback type: receives a list of clarifying questions, returns a string
# with the user's answers.  Can be sync or async.
HumanInputFn = Callable[[list[str]], str | Awaitable[str]]


# ---------------------------------------------------------------------------
# QuerySession — stateful, multi-turn session for a single user query
# ---------------------------------------------------------------------------

class QuerySession:
    """
    A stateful session that stores a query once and manages multi-round
    human-in-the-loop clarification before routing.

    Usage::

        session = network.create_session("What are the tax penalties for late filing?")

        # Optionally inspect what context is missing before running
        print(await session.pending_questions_async())

        # Run with a human input callback — may ask multiple rounds
        result = await session.run_async(human_input_fn=my_callback)
        print(result.answer)

        # The full conversation (clarifying Q&A + final answer) is in .history
        for turn in session.history:
            print(turn["role"], ":", turn["content"])
    """

    def __init__(self, network: "AgentNetwork", query: str) -> None:
        self.original_query: str = query
        # Accumulated conversation: each turn is {"role": "assistant"|"user", "content": str}
        self.history: list[dict[str, str]] = [{"role": "user", "content": query}]
        self.clarifications_asked: list[str] = []
        self._network = network
        self._enriched_query: str = query  # grows as context is gathered

    # ------------------------------------------------------------------
    # Public run API
    # ------------------------------------------------------------------

    async def run_async(
        self,
        human_input_fn: Optional[HumanInputFn] = None,
        synthesize: bool = True,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
        max_clarification_rounds: int = 3,
    ) -> "NetworkQueryResult":
        """
        Run the stored query through the network.

        Parameters
        ----------
        human_input_fn:
            Sync or async callable ``(questions: list[str]) -> str``.
            Called once per clarification round when critical context is missing.
            If omitted the agents answer (or request clarification) with whatever
            context is already in the query.
        synthesize:
            Merge all agent responses into one final answer (default True).
        max_clarification_rounds:
            Maximum number of back-and-forth rounds with the human before routing.
        """
        if human_input_fn is not None and self._network._provider is not None:
            await self._gather_context_iterative(human_input_fn, max_clarification_rounds)

        if broadcast:
            result = await self._network._broadcast(self._enriched_query)
        else:
            entry = self._network._resolve_entry(entry_agent)
            result = await self._network._route(self._enriched_query, entry)

        result.query = self.original_query
        result.enriched_query = self._enriched_query
        result.clarifications_asked = list(self.clarifications_asked)

        if synthesize and len(result.all_responses) > 1 and self._network._provider:
            synthesised = await self._network._synthesise_final_answer(
                self._enriched_query, result.all_responses
            )
            final = AgentResponse(agent_name="Synthesiser", content=synthesised)
            result.primary_response = final
            result.all_responses.append(final)

        self.history.append({"role": "assistant", "content": result.answer})
        return result

    def run(
        self,
        human_input_fn: Optional[HumanInputFn] = None,
        synthesize: bool = True,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
        max_clarification_rounds: int = 3,
    ) -> "NetworkQueryResult":
        """Synchronous wrapper around run_async."""
        return asyncio.run(
            self.run_async(
                human_input_fn=human_input_fn,
                synthesize=synthesize,
                entry_agent=entry_agent,
                broadcast=broadcast,
                max_clarification_rounds=max_clarification_rounds,
            )
        )

    async def pending_questions_async(self) -> list[str]:
        """Return the clarifying questions the network would ask right now."""
        missing = await self._network._detect_missing_context(self._enriched_query)
        return [m["question"] for m in missing if "question" in m]

    def pending_questions(self) -> list[str]:
        """Synchronous wrapper around pending_questions_async."""
        return asyncio.run(self.pending_questions_async())

    # ------------------------------------------------------------------
    # Multi-round context gathering
    # ------------------------------------------------------------------

    async def _gather_context_iterative(
        self, human_input_fn: HumanInputFn, max_rounds: int
    ) -> None:
        """
        Iteratively ask for missing context until the network is satisfied
        or max_rounds is reached.  Each round appends to self.history.
        """
        for round_num in range(max_rounds):
            missing = await self._network._detect_missing_context(self._enriched_query)
            if not missing:
                break

            questions = [m["question"] for m in missing if "question" in m]
            if not questions:
                break

            self.clarifications_asked.extend(questions)
            self.history.append({
                "role": "assistant",
                "content": "Clarification needed:\n" + "\n".join(f"• {q}" for q in questions),
            })

            logger.info(f"[Session] Round {round_num + 1}: asking {len(questions)} question(s).")

            if asyncio.iscoroutinefunction(human_input_fn):
                answers = await human_input_fn(questions)
            else:
                answers = human_input_fn(questions)

            self.history.append({"role": "user", "content": answers})
            self._enriched_query += f"\n\nContext (round {round_num + 1}):\n{answers}"

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QuerySession("
            f"query={self.original_query[:60]!r}{'...' if len(self.original_query) > 60 else ''}, "
            f"history_turns={len(self.history)}, "
            f"clarifications={len(self.clarifications_asked)})"
        )

# ---------------------------------------------------------------------------
# Context-gap analyser prompt
# ---------------------------------------------------------------------------

_CONTEXT_ANALYSER_SYSTEM = """\
You are a query pre-processor for a multi-agent system.
Your ONLY job is to identify critical missing context variables that would
SIGNIFICANTLY change the answer to the user's question.

Think domain-specifically:
- Legal/compliance → jurisdiction (country, state, province), governing law, date of incident
- Medical/health    → country/healthcare system, patient age, existing conditions
- Financial/tax     → country/region, currency, fiscal year, entity type
- Technical/code    → OS, language version, framework version, deployment environment
- Education/admission → country, institution type, academic year

Return ONLY a JSON object in this exact format:
{
  "missing": [
    {"variable": "<name>", "question": "<specific question to ask the user>"}
  ],
  "can_proceed": <true|false>
}

Set "can_proceed" to true only when no critical context is missing.
Return ONLY the JSON — no commentary, no markdown fences.
"""

_CONTEXT_ANALYSER_PROMPT = """\
User query: {query}

Identify any critical missing context. Return only the JSON.
"""

# ---------------------------------------------------------------------------
# Response synthesiser prompt
# ---------------------------------------------------------------------------

_SYNTHESISER_SYSTEM = """\
You are a response synthesiser for a multi-agent system.
You receive the outputs from one or more specialist agents and the original
user query.  Produce ONE clear, coherent final answer.

Rules:
1. Merge all relevant information — do not repeat the same point twice.
2. Speak directly to the user; do not say "Agent X said…" unless attribution
   genuinely adds clarity.
3. If agents gave conflicting information, acknowledge the conflict briefly
   and explain the distinction (e.g. different jurisdictions).
4. Be concise — shorter is better when completeness is maintained.
5. Do NOT add information beyond what the agents provided.
"""

_SYNTHESISER_PROMPT = """\
Original user query:
{query}

Specialist agent responses:
{responses}

Synthesise these into one final answer for the user.
"""


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class NetworkQueryResult:
    """Aggregated result of running a query through the network."""

    def __init__(
        self,
        query: str,
        primary_response: AgentResponse,
        all_responses: list[AgentResponse],
        routed_path: list[str],
        enriched_query: str = "",
        clarifications_asked: list[str] | None = None,
    ) -> None:
        self.query = query
        self.enriched_query = enriched_query or query
        self.primary_response = primary_response
        self.all_responses = all_responses
        self.routed_path = routed_path
        self.clarifications_asked: list[str] = clarifications_asked or []

    @property
    def answer(self) -> str:
        return self.primary_response.content

    def __str__(self) -> str:
        return self.answer

    def __repr__(self) -> str:
        return (
            f"NetworkQueryResult(agents_involved={self.routed_path}, "
            f"answer_length={len(self.answer)})"
        )

    def full_report(self) -> str:
        """Return a formatted multi-agent response report."""
        lines = [
            f"Query: {self.query}",
            f"Routing path: {' → '.join(self.routed_path)}",
            "",
        ]
        if self.clarifications_asked:
            lines.append("Clarifications gathered:")
            for q in self.clarifications_asked:
                lines.append(f"  • {q}")
            lines.append("")
        for resp in self.all_responses:
            lines.append(f"[{resp.agent_name}]")
            lines.append(resp.content)
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main network class
# ---------------------------------------------------------------------------


class AgentNetwork:
    """
    The live multi-agent network synthesized by ChorusAgents.

    Holds the agent instances, the communication graph, and the routing
    logic that dispatches a user query to the appropriate agent(s).
    """

    def __init__(
        self,
        blueprint: NetworkBlueprint,
        agents: dict[str, Agent],
        provider: Any = None,
    ) -> None:
        self.blueprint = blueprint
        self.agents = agents
        self._graph = self._build_graph(blueprint)
        # Provider used for meta-operations (context analysis, synthesis).
        # Falls back to the first available agent's provider.
        self._provider = provider or self._first_provider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def meta_role(self) -> str:
        return self.blueprint.meta_role

    @property
    def topology(self) -> TopologyType:
        return self.blueprint.topology_type

    @property
    def agent_names(self) -> list[str]:
        return list(self.agents.keys())

    def get_agent(self, name: str) -> Agent:
        if name not in self.agents:
            raise KeyError(f"No agent named {name!r} in this network.")
        return self.agents[name]

    def create_session(self, query: str) -> QuerySession:
        """
        Create a stateful QuerySession for the given query.

        The query is stored once in the session.  Call session.run() to
        execute it — with optional human-in-the-loop clarification and
        multi-round context gathering.

        Example::

            session = network.create_session("What are penalties for tax evasion?")
            result  = session.run(human_input_fn=my_callback)
            print(result.answer)
            print(session.history)   # full Q&A transcript
        """
        return QuerySession(network=self, query=query)

    async def query_async(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
        human_input_fn: Optional[HumanInputFn] = None,
        synthesize: bool = True,
    ) -> NetworkQueryResult:
        """
        Route a user query through the agent network.

        Parameters
        ----------
        user_query:
            The natural-language question or task.
        entry_agent:
            Name of the agent to handle the query first.
            Defaults to hub (star) or first (pipeline/others).
        broadcast:
            If True, all agents receive the query and responses are aggregated.
        human_input_fn:
            Optional async or sync callable that receives a list of clarifying
            questions and returns the user's answers as a string.
            When provided, the network will pause before routing to ask the
            user for any critical missing context (jurisdiction, etc.).
        synthesize:
            If True (default), agent responses are synthesised into one final
            answer by the meta-provider.  Set False to get raw agent outputs.
        """
        clarifications_asked: list[str] = []
        enriched_query = user_query

        # Step 1 — Gather missing context (human-in-the-loop)
        if human_input_fn is not None and self._provider is not None:
            enriched_query, clarifications_asked = await self._gather_missing_context(
                user_query, human_input_fn
            )

        # Step 2 — Route through agents
        if broadcast:
            result = await self._broadcast(enriched_query)
        else:
            entry = self._resolve_entry(entry_agent)
            result = await self._route(enriched_query, entry)

        result.query = user_query
        result.enriched_query = enriched_query
        result.clarifications_asked = clarifications_asked

        # Step 3 — Synthesise into ONE final answer
        if synthesize and len(result.all_responses) > 1 and self._provider is not None:
            synthesised_text = await self._synthesise_final_answer(
                enriched_query, result.all_responses
            )
            final_response = AgentResponse(
                agent_name="Synthesiser",
                content=synthesised_text,
                routed_path=result.routed_path,
            )
            result.primary_response = final_response
            result.all_responses.append(final_response)

        return result

    def query(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
        human_input_fn: Optional[HumanInputFn] = None,
        synthesize: bool = True,
    ) -> NetworkQueryResult:
        """Synchronous wrapper around query_async."""
        return asyncio.run(
            self.query_async(
                user_query,
                entry_agent=entry_agent,
                broadcast=broadcast,
                human_input_fn=human_input_fn,
                synthesize=synthesize,
            )
        )

    def describe(self) -> str:
        """Return a human-readable summary of the network."""
        lines = [
            f"Network: {self.meta_role}",
            f"Topology: {self.topology.value}",
            f"Agents ({len(self.agents)}):",
        ]
        for agent in self.agents.values():
            lines.append(f"  - {agent.name}: {agent.sub_role}")
            if agent.neighbors:
                lines.append(f"      connects to: {', '.join(agent.neighbors)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context-gap detection & human-in-the-loop
    # ------------------------------------------------------------------

    async def _detect_missing_context(self, query: str) -> list[dict]:
        """
        Ask the LLM which critical context variables are missing from query.
        Returns a list of {"variable": ..., "question": ...} dicts, or []
        if the query has enough context to proceed.
        """
        if self._provider is None:
            return []
        prompt = _CONTEXT_ANALYSER_PROMPT.format(query=query)
        try:
            raw = await self._provider.complete(
                system=_CONTEXT_ANALYSER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            data = self._parse_json_safe(raw)
        except Exception as exc:
            logger.warning(f"Context analyser failed: {exc}.")
            return []
        if data.get("can_proceed", True):
            return []
        return data.get("missing", [])

    async def _gather_missing_context(
        self,
        query: str,
        human_input_fn: HumanInputFn,
    ) -> tuple[str, list[str]]:
        """
        Single-round context gathering used by query_async().
        Delegates detection to _detect_missing_context, then calls human_input_fn once.
        For multi-round gathering, use QuerySession instead.
        """
        missing = await self._detect_missing_context(query)
        if not missing:
            return query, []

        questions = [item["question"] for item in missing if "question" in item]
        if not questions:
            return query, []

        logger.info(f"Asking user {len(questions)} clarifying question(s).")

        if asyncio.iscoroutinefunction(human_input_fn):
            user_answers = await human_input_fn(questions)
        else:
            user_answers = human_input_fn(questions)

        enriched = f"{query}\n\nAdditional context provided by the user:\n{user_answers}"
        return enriched, questions

    # ------------------------------------------------------------------
    # Response synthesis
    # ------------------------------------------------------------------

    async def _synthesise_final_answer(
        self,
        query: str,
        responses: list[AgentResponse],
    ) -> str:
        """Combine multiple agent responses into one coherent answer."""
        formatted = "\n\n".join(
            f"[{r.agent_name}]\n{r.content}" for r in responses
        )
        prompt = _SYNTHESISER_PROMPT.format(query=query, responses=formatted)
        try:
            return await self._provider.complete(
                system=_SYNTHESISER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.warning(f"Synthesis failed: {exc}. Returning last agent response.")
            return responses[-1].content

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    async def _route(self, query: str, entry_agent_name: str) -> NetworkQueryResult:
        """Route through the network starting from entry_agent."""
        routed_path: list[str] = []
        all_responses: list[AgentResponse] = []

        current_name = entry_agent_name
        visited: set[str] = set()

        while current_name and current_name not in visited:
            visited.add(current_name)
            routed_path.append(current_name)

            agent = self.agents[current_name]
            message = AgentMessage(
                sender="user" if len(routed_path) == 1 else routed_path[-2],
                content=query,
            )
            response = await agent.process(message)
            all_responses.append(response)

            next_agent = self._detect_handoff(response.content, agent.neighbors, visited)
            current_name = next_agent

        primary = all_responses[-1] if all_responses else AgentResponse(
            agent_name="network", content="No agents processed the query."
        )
        return NetworkQueryResult(
            query=query,
            primary_response=primary,
            all_responses=all_responses,
            routed_path=routed_path,
        )

    async def _broadcast(self, query: str) -> NetworkQueryResult:
        """Send the query to every agent in parallel and aggregate results."""
        tasks = []
        for name, agent in self.agents.items():
            msg = AgentMessage(sender="user", content=query)
            tasks.append(agent.process(msg))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[AgentResponse] = []
        for name, result in zip(self.agents.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"Agent {name!r} failed during broadcast: {result}")
                responses.append(AgentResponse(
                    agent_name=name,
                    content=f"[Error: {result}]",
                ))
            else:
                responses.append(result)

        combined = "\n\n".join(
            f"[{r.agent_name}]\n{r.content}" for r in responses
        )
        primary = AgentResponse(
            agent_name="network_broadcast",
            content=combined,
        )
        return NetworkQueryResult(
            query=query,
            primary_response=primary,
            all_responses=responses,
            routed_path=[r.agent_name for r in responses],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_entry(self, entry_agent: Optional[str]) -> str:
        if entry_agent:
            if entry_agent not in self.agents:
                raise KeyError(f"Entry agent {entry_agent!r} not found in network.")
            return entry_agent

        if self.topology == TopologyType.STAR:
            return self._find_hub()

        if self.topology == TopologyType.PIPELINE:
            return self._pipeline_start()

        return sorted(self.agents.keys())[0]

    def _find_hub(self) -> str:
        """Return the highest-degree node (the star's center)."""
        if not self.agents:
            raise RuntimeError("Cannot find hub: network has no agents.")
        degrees = list(self._graph.degree())
        if not degrees:
            return next(iter(self.agents))
        return max(degrees, key=lambda x: x[1])[0]

    def _pipeline_start(self) -> str:
        """Return the node with in-degree 0 (pipeline source)."""
        for node, in_deg in self._graph.in_degree():
            if in_deg == 0:
                return node
        return next(iter(self.agents))

    def _first_provider(self) -> Any:
        """Return the provider from the first available agent."""
        for agent in self.agents.values():
            if hasattr(agent, "provider") and agent.provider is not None:
                return agent.provider
        return None

    @staticmethod
    def _detect_handoff(
        response_text: str,
        neighbors: list[str],
        visited: set[str],
    ) -> Optional[str]:
        """
        Detect if an agent response explicitly routes to a neighbor.
        Uses word-boundary matching to avoid false positives.
        """
        for neighbor in neighbors:
            if neighbor not in visited:
                pattern = r"\b" + re.escape(neighbor) + r"\b"
                if re.search(pattern, response_text, re.IGNORECASE):
                    return neighbor
        return None

    @staticmethod
    def _parse_json_safe(text: str) -> dict:
        """Parse JSON from LLM output, stripping markdown fences if present."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            if start != -1:
                depth = 0
                for i, ch in enumerate(text[start:], start=start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start: i + 1])
                            except json.JSONDecodeError:
                                break
        return {}

    @staticmethod
    def _build_graph(blueprint: NetworkBlueprint) -> nx.DiGraph:
        G = nx.DiGraph()
        for agent in blueprint.agents:
            G.add_node(agent.name, sub_role=agent.sub_role)
        for edge in blueprint.edges:
            G.add_edge(edge.source, edge.target, label=edge.label)
            if edge.direction == EdgeDirection.BIDIRECTIONAL:
                G.add_edge(edge.target, edge.source, label=edge.label)
        return G

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
