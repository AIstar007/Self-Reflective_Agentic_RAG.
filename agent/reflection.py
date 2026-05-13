import json
from dataclasses import dataclass


@dataclass
class ReflectionResult:
    is_sufficient: bool
    missing_points: str
    needs_more_retrieval: bool
    refined_query: str


REFLECTION_PROMPT_TEMPLATE = """
You are a strict evaluator for a retrieval-augmented QA system.
Question:
{question}

Retrieved Context Summary:
{context_summary}

Draft Answer:
{draft_answer}

Evaluate:
1) Is the answer complete and directly supported by retrieved context?
2) Is more retrieval needed?
3) What is missing or ambiguous?
4) If retrieval is needed, provide a better refined query.

Return ONLY valid JSON with this schema:
{{
  "is_sufficient": true/false,
  "missing_points": "...",
  "needs_more_retrieval": true/false,
  "refined_query": "..."
}}
""".strip()


def parse_reflection_json(text: str, original_query: str) -> ReflectionResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ReflectionResult(
            is_sufficient=False,
            missing_points="Failed to parse reflection JSON.",
            needs_more_retrieval=True,
            refined_query=original_query,
        )

    return ReflectionResult(
        is_sufficient=bool(payload.get("is_sufficient", False)),
        missing_points=str(payload.get("missing_points", "")).strip(),
        needs_more_retrieval=bool(payload.get("needs_more_retrieval", True)),
        refined_query=str(payload.get("refined_query", original_query)).strip() or original_query,
    )
