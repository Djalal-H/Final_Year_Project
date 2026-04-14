"""
Module 10: PromptBuilder

Transforms a structured report into the final LLM prompt string,
using template variants selected by decision class.

System prompt constraints (from the specification):
    - Only reference information present in the report — no speculation
    - Never claim to know what the model "might" have been thinking
    - Cap at 3–5 sentences
    - Always cover: (1) what the vehicle did and why,
      (2) what the most dangerous alternative would have caused,
      (3) whether attention confirms the decision
"""

from typing import Any, Dict, Tuple

from xai.narration.narration_router import BRIEF, DETAILED, DETAILED_CAVEAT


# =============================================================================
# SHARED SYSTEM PROMPT
# =============================================================================

_SYSTEM_PROMPT = (
    "You are the explanatory module for an Autonomous Driving RL Agent. "
    "Your role is to narrate the agent's decision using ONLY the structured "
    "report provided. Rules:\n"
    "1. Only reference information present in the report — do not speculate.\n"
    "2. Never claim to know what the model 'might' have been thinking.\n"
    "3. Keep your response to 3–5 sentences.\n"
    "4. Always cover: (a) what the vehicle did and why, "
    "(b) what the most dangerous alternative would have caused, "
    "(c) whether attention confirms the decision."
)


# =============================================================================
# TEMPLATE BUILDERS
# =============================================================================


def _describe_nearby_agents(report: Dict[str, Any]) -> str:
    """Convert the scene_description dictionary into readable sentences for the LLM."""
    desc_list = report.get("scene_description", [])
    if not desc_list:
        return "No agents nearby."

    lines = ["Nearby agents:"]
    for d in desc_list:
        agent_id = d.get("agent_id")
        dist = d.get("distance", "?")
        rels = d.get("relations", [])
        ttc = d.get("ttc")

        # Format relations into a readable list
        rel_str = ", ".join(rels).replace("_", " ") if rels else "no specific relation"
        
        sentence = f"- Agent {agent_id} is {dist}m away ({rel_str})"
        if ttc is not None:
            sentence += f" with TTC {ttc}s"
        sentence += "."
        lines.append(sentence)

    return " ".join(lines)


def _build_detailed(report: Dict[str, Any]) -> str:
    """Full causal explanation for GROUNDED_CRITICAL."""
    ego = report["ego_state"]
    chosen = report["chosen_action"]
    alts = report.get("alternatives", [])
    grounding = report.get("attention_grounding", {})
    necessity = report.get("necessity_score", 0)

    scene_str = _describe_nearby_agents(report)

    lines = [
        f"The Ego Vehicle is {ego.get('ego_action', 'moving')} "
        f"at {ego.get('ego_velocity', '?')} m/s.",
        scene_str,
        f"The agent chose: {chosen.get('label', 'unknown action')}.",
        f"Decision necessity: {necessity:.0%} of alternatives would fail.",
    ]

    # Most dangerous alternative
    collisions = [a for a in alts if a.get("outcome") == "COLLISION"]
    if collisions:
        worst = min(collisions, key=lambda a: a.get("min_ttc") or 999)
        lines.append(
            f"Most dangerous alternative: \"{worst.get('label')}\" → "
            f"COLLISION with Agent {worst.get('threat_agent_id')} "
            f"in {worst.get('min_ttc', '?')}s."
        )

    # Grounding confirmation
    g_score = grounding.get("grounding_score")
    if g_score is not None:
        breakdown = grounding.get("per_agent_breakdown", [])
        top_agents = sorted(
            breakdown, key=lambda x: x.get("attention_mass", 0), reverse=True
        )[:3]
        agent_desc = ", ".join(
            f"Agent {a['agent_id']} ({a['attention_mass']:.1%})"
            for a in top_agents
        )
        lines.append(
            f"Attention grounding score: {g_score:.2f}. "
            f"Top attended threat agents: {agent_desc}."
        )

    return "\n".join(lines)


def _build_detailed_with_caveat(report: Dict[str, Any]) -> str:
    """Full causal explanation + transparency caveat for UNGROUNDED_CRITICAL."""
    base = _build_detailed(report)

    caveat = (
        "\n⚠️ TRANSPARENCY CAVEAT: Although the agent avoided danger, "
        "its attention distribution does NOT confirm awareness of the "
        "identified threat agents. This decision is flagged for further "
        "review — the safe outcome may be coincidental rather than "
        "intentional."
    )
    return base + caveat


def _build_brief(report: Dict[str, Any]) -> str:
    """Brief narration for GROUNDED_ROUTINE / ROUTINE."""
    ego = report["ego_state"]
    chosen = report["chosen_action"]
    necessity = report.get("necessity_score", 0)
    decision_class = report.get("decision_class", "ROUTINE")

    scene_str = _describe_nearby_agents(report)

    lines = [
        f"The Ego Vehicle is {ego.get('ego_action', 'moving')} "
        f"at {ego.get('ego_velocity', '?')} m/s.",
        scene_str,
        f"Action: {chosen.get('label', 'unknown')}. "
        f"Decision class: {decision_class} "
        f"(necessity {necessity:.0%}).",
        "No critical threats detected; routine driving conditions.",
    ]
    return "\n".join(lines)


# =============================================================================
# PUBLIC API
# =============================================================================

_TEMPLATE_MAP = {
    DETAILED: _build_detailed,
    DETAILED_CAVEAT: _build_detailed_with_caveat,
    BRIEF: _build_brief,
}


def build_prompt(
    template_key: str,
    report: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build the ``(system_prompt, user_prompt)`` pair for the LLM.

    Args:
        template_key: One of ``"detailed"``, ``"detailed_caveat"``, ``"brief"``.
        report: Structured report dict.

    Returns:
        Tuple of ``(system_prompt, user_prompt)`` strings.
    """
    builder = _TEMPLATE_MAP.get(template_key, _build_brief)
    user_prompt = builder(report)
    return _SYSTEM_PROMPT, user_prompt
