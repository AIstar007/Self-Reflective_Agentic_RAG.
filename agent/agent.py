import logging
from typing import Any, Dict, List, Optional

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from agent.reflection import REFLECTION_PROMPT_TEMPLATE, ReflectionResult, parse_reflection_json
from agent.tools import RetrievalTool, extract_relevant_paragraphs
from config import Settings

logger = logging.getLogger(__name__)


ANSWER_PROMPT_TEMPLATE = """
You are an enterprise assistant answering based only on provided retrieved context.
If context is insufficient, explicitly say what is missing.

Question:
{question}

Retrieved Context:
{context}

Provide a concise and factual answer.
""".strip()


FLIGHT_FALLBACK_PROMPT_TEMPLATE = """
You are an enterprise assistant.
The internal indexed documents were either missing or weakly matched for this question.

Question:
{question}

Answer using your general knowledge.
Rules:
- Be practical and concise.
- If details vary by provider/country, state that clearly.
- Do not claim that this answer came from indexed documents.
""".strip()


class SelfReflectiveRAGAgent:
    def __init__(self, settings: Settings, retrieval_tool: RetrievalTool) -> None:
        self._settings = settings
        self._retrieval_tool = retrieval_tool

        self._kernel = Kernel()
        self._service_id = "azure-chat"
        self._kernel.add_service(
            AzureChatCompletion(
                service_id=self._service_id,
                deployment_name=settings.azure_openai_chat_deployment,
                endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
        )

    async def ask(
        self,
        question: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_paragraph_focus: bool = True,
    ) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        working_query = question
        iterations: List[Dict[str, Any]] = []
        last_document_answer: Optional[str] = None

        for round_idx in range(1, self._settings.max_reflection_rounds + 1):
            logger.info("RAG loop iteration %s for query: %s", round_idx, working_query)
            retrieved = self._retrieval_tool.retrieve(
                query=working_query,
                top_k=self._settings.retrieval_top_k,
                metadata_filter=metadata_filter,
            )

            if not retrieved:
                logger.warning("No retrieval results found")
                fallback_answer = await self._answer_from_external_knowledge(question)
                iterations.append(
                    {
                        "round": round_idx,
                        "query": working_query,
                        "retrieved_count": 0,
                        "answer": fallback_answer,
                        "reflection": {
                            "is_sufficient": True,
                            "missing_points": "No indexed support found; used external LLM fallback.",
                            "needs_more_retrieval": False,
                            "refined_query": working_query,
                        },
                    }
                )
                return {
                    "answer": fallback_answer,
                    "iterations": iterations,
                    "final_query": working_query,
                    "answer_mode": "external_llm",
                }

            if self._is_weak_match(retrieved):
                logger.info("Weak retrieval match detected; using external LLM fallback")
                fallback_answer = await self._answer_from_external_knowledge(question)
                iterations.append(
                    {
                        "round": round_idx,
                        "query": working_query,
                        "retrieved_count": len(retrieved),
                        "answer": fallback_answer,
                        "reflection": {
                            "is_sufficient": True,
                            "missing_points": "Weak document match; used external LLM fallback.",
                            "needs_more_retrieval": False,
                            "refined_query": working_query,
                        },
                    }
                )
                return {
                    "answer": fallback_answer,
                    "iterations": iterations,
                    "final_query": working_query,
                    "answer_mode": "external_llm",
                }

            focused = extract_relevant_paragraphs(working_query, retrieved) if use_paragraph_focus else retrieved
            context_text = self._format_context(focused)
            answer = await self._complete(
                ANSWER_PROMPT_TEMPLATE.format(question=question, context=context_text)
            )
            last_document_answer = answer

            if self._is_document_non_answer(answer):
                fallback_answer = await self._answer_from_external_knowledge(question)
                iterations.append(
                    {
                        "round": round_idx,
                        "query": working_query,
                        "retrieved_count": len(retrieved),
                        "answer": fallback_answer,
                        "reflection": {
                            "is_sufficient": True,
                            "missing_points": "Document context did not contain the requested fact; used external LLM fallback.",
                            "needs_more_retrieval": False,
                            "refined_query": working_query,
                        },
                    }
                )
                return {
                    "answer": fallback_answer,
                    "iterations": iterations,
                    "final_query": working_query,
                    "answer_mode": "external_llm",
                }

            reflection = await self._reflect(question, focused, answer)
            iterations.append(
                {
                    "round": round_idx,
                    "query": working_query,
                    "retrieved_count": len(retrieved),
                    "answer": answer,
                    "reflection": reflection.__dict__,
                }
            )

            if reflection.is_sufficient:
                return {
                    "answer": answer,
                    "iterations": iterations,
                    "final_query": working_query,
                    "answer_mode": "document",
                }

            if not reflection.needs_more_retrieval:
                if last_document_answer:
                    return {
                        "answer": last_document_answer,
                        "iterations": iterations,
                        "final_query": working_query,
                        "answer_mode": "document",
                    }

                fallback_answer = await self._answer_from_external_knowledge(question)
                iterations.append(
                    {
                        "round": round_idx,
                        "query": working_query,
                        "retrieved_count": len(retrieved),
                        "answer": fallback_answer,
                        "reflection": {
                            "is_sufficient": True,
                            "missing_points": "Retrieved context remained insufficient; used external LLM fallback.",
                            "needs_more_retrieval": False,
                            "refined_query": working_query,
                        },
                    }
                )
                return {
                    "answer": fallback_answer,
                    "iterations": iterations,
                    "final_query": working_query,
                    "answer_mode": "external_llm",
                }

            working_query = reflection.refined_query

        if last_document_answer:
            return {
                "answer": last_document_answer,
                "iterations": iterations,
                "final_query": working_query,
                "answer_mode": "document",
            }

        fallback_answer = await self._answer_from_external_knowledge(question)
        iterations.append(
            {
                "round": self._settings.max_reflection_rounds + 1,
                "query": working_query,
                "retrieved_count": 0,
                "answer": fallback_answer,
                "reflection": {
                    "is_sufficient": True,
                    "missing_points": "Exceeded reflection rounds with insufficient context; used external LLM fallback.",
                    "needs_more_retrieval": False,
                    "refined_query": working_query,
                },
            }
        )
        return {
            "answer": fallback_answer,
            "iterations": iterations,
            "final_query": working_query,
            "answer_mode": "external_llm",
        }

    async def _reflect(
        self,
        question: str,
        context_units: List[Dict[str, Any]],
        draft_answer: str,
    ) -> ReflectionResult:
        context_summary = "\n".join(
            [
                f"- {item.get('metadata', {}).get('source')} | "
                f"{item.get('metadata', {}).get('unit_type')} "
                f"{item.get('metadata', {}).get('index')}"
                for item in context_units
            ]
        )

        reflection_raw = await self._complete(
            REFLECTION_PROMPT_TEMPLATE.format(
                question=question,
                context_summary=context_summary,
                draft_answer=draft_answer,
            )
        )
        return parse_reflection_json(reflection_raw, question)

    async def _complete(self, prompt: str) -> str:
        result = await self._kernel.invoke_prompt(
            prompt,
            service_id=self._service_id,
        )
        return str(result).strip()

    async def _answer_from_external_knowledge(self, question: str) -> str:
        return await self._complete(FLIGHT_FALLBACK_PROMPT_TEMPLATE.format(question=question))

    def _is_weak_match(self, retrieved_units: List[Dict[str, Any]]) -> bool:
        distances = []
        for item in retrieved_units:
            distance = item.get("distance")
            if isinstance(distance, (int, float)):
                distances.append(float(distance))

        if not distances:
            return False

        best_distance = min(distances)
        logger.info(
            "Best retrieval distance: %.4f (weak threshold: %.4f)",
            best_distance,
            self._settings.retrieval_weak_match_distance,
        )
        return best_distance > self._settings.retrieval_weak_match_distance

    @staticmethod
    def _is_document_non_answer(answer: str) -> bool:
        text = answer.lower()
        markers = [
            "does not mention",
            "cannot confirm",
            "can't confirm",
            "insufficient context",
            "insufficient information",
            "not enough information",
            "not provided in the context",
            "based on this context alone",
            "context does not contain",
            "information that is missing",
        ]
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_flight_related(question: str) -> bool:
        terms = {
            "flight",
            "airline",
            "airport",
            "boarding",
            "baggage",
            "luggage",
            "check-in",
            "checkin",
            "ticket",
            "pnr",
            "terminal",
            "departure",
            "arrival",
            "layover",
            "connecting",
            "cabin",
            "gate",
            "infant",
            "refund",
            "reschedule",
            "cancel",
            "no-show",
            "aviation",
            "fare",
            "seat",
            "upgrade",
            "meal",
            "web check-in",
            "web checkin",
            "domestic",
            "international",
        }
        q = question.lower()
        return any(term in q for term in terms)

    @staticmethod
    def _format_context(retrieved_units: List[Dict[str, Any]]) -> str:
        blocks = []
        for item in retrieved_units:
            metadata = item.get("metadata", {})
            blocks.append(
                "\n".join(
                    [
                        f"Source: {metadata.get('source')}",
                        f"Unit Type: {metadata.get('unit_type')}",
                        f"Index: {metadata.get('index')}",
                        "Content:",
                        str(item.get("content", "")),
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)
