"""Domain models (pydantic v2): LLM output schemas and the immutable, serializable agent state."""
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    """Strict base: unknown fields from the LLM are errors, not silently dropped."""
    model_config = ConfigDict(extra="forbid")


# ---- Structured outputs produced by the LLM ---------------------------------------
class Plan(_Model):
    steps: list[str]


class Hypothesis(_Model):
    statement: str
    rationale: str
    confidence: float


class HypothesisSet(_Model):
    hypotheses: list[Hypothesis]


class Variable(_Model):
    name: str
    low: float
    high: float


class Claim(_Model):
    """A machine-checkable formalization of a hypothesis.

    Expressions use a restricted math DSL (see tools.safe_eval). ``definitions`` are
    evaluated in order, so later ones may reference earlier ones.
    """
    id: str
    description: str
    kind: Literal["identity", "monte_carlo"]
    lhs: str
    rhs: str = "0"
    definitions: dict[str, str] = Field(default_factory=dict)
    variables: list[Variable] = Field(default_factory=list)   # sampling domains
    params: dict[str, float] = Field(default_factory=dict)    # fixed values (monte_carlo)


class Formalization(_Model):
    claims: list[Claim]


class Critique(_Model):
    verdict: Literal["accept", "revise"]
    feedback: str
    failed_claim_ids: list[str]


class Report(_Model):
    title: str
    summary: str
    findings: list[str]
    limitations: list[str]


# ---- Tool-side results ---------------------------------------------------------------
class ClaimResult(_Model):
    claim_id: str
    status: Literal["verified_symbolic", "verified_numeric", "refuted", "error"]
    detail: str
    counterexample: Optional[dict[str, float]] = None

    @property
    def verified(self) -> bool:
        return self.status.startswith("verified")


# ---- Agent state -----------------------------------------------------------------------
class AgentState(_Model):
    """Immutable (frozen) and JSON-serializable: checkpoint, resume and audit for free."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    problem: str
    max_iterations: int = 3
    iteration: int = 0
    status: Literal["running", "accepted", "exhausted"] = "running"
    plan: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    results: list[ClaimResult] = Field(default_factory=list)
    critique: Optional[Critique] = None
    feedback_log: list[str] = Field(default_factory=list)
    report_md: str = ""

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        return cls.model_validate(data)
