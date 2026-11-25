"""Structured output generation: pydantic validation + self-repair loop."""
import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLM

T = TypeVar("T", bound=BaseModel)


class OutputParseError(ValueError):
    pass


class StructuredOutputError(RuntimeError):
    pass


def extract_json(text: str):
    """Find the first decodable JSON value in free-form model output (handles code fences)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise OutputParseError("no JSON value found in model output")


def format_errors(exc: ValidationError) -> str:
    """Compact, model-readable error list: 'claims.1.kind: Input should be ...'."""
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '$'}: {e['msg']}" for e in exc.errors())


def generate(llm: LLM, role: str, schema: type[T], system: str, user: str, retries: int = 2) -> T:
    """Ask the model for ``schema``; on invalid output, feed the exact errors back and retry."""
    prompt = (f"{user}\n\nRespond with ONLY a JSON object matching this JSON Schema:\n"
              f"{json.dumps(schema.model_json_schema())}")
    feedback, last_error = "", ""
    for _ in range(retries + 1):
        raw = llm.complete(role, system, prompt + feedback)
        try:
            return schema.model_validate(extract_json(raw))
        except OutputParseError as exc:
            last_error = str(exc)
        except ValidationError as exc:
            last_error = format_errors(exc)
        feedback = f"\n\nYour previous reply was invalid: {last_error}. Return only corrected JSON."
    raise StructuredOutputError(f"{role}: no valid {schema.__name__} after {retries + 1} attempts: {last_error}")
