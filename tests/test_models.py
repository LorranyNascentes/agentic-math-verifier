import unittest

from pydantic import ValidationError

from stem_agents.models import AgentState, Claim, Critique
from stem_agents.structured import format_errors

BASE = {"id": "a", "description": "d", "kind": "identity", "lhs": "x"}


class ModelTests(unittest.TestCase):
    def test_valid_claim_with_defaults(self):
        c = Claim.model_validate({**BASE, "rhs": "x", "variables": [{"name": "x", "low": 0, "high": 1}]})
        self.assertEqual(c.variables[0].high, 1.0)
        self.assertEqual(c.definitions, {})

    def test_errors_carry_field_paths_for_llm_repair(self):
        with self.assertRaises(ValidationError) as cm:
            Claim.model_validate({**BASE, "kind": "bogus"})
        self.assertIn("kind", format_errors(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            Claim.model_validate({**BASE, "variables": [{"name": "x", "low": "zero", "high": 1}]})
        self.assertIn("variables.0.low", format_errors(cm.exception))

    def test_strict_unknown_and_missing_fields(self):
        with self.assertRaises(ValidationError):
            Critique.model_validate({"verdict": "accept", "feedback": "", "failed_claim_ids": [], "extra": 1})
        with self.assertRaises(ValidationError):
            Critique.model_validate({"verdict": "accept"})

    def test_json_schema_for_prompting(self):
        self.assertIn("kind", Claim.model_json_schema()["properties"])

    def test_state_is_immutable_and_roundtrips(self):
        s = AgentState(problem="p")
        self.assertEqual(AgentState.from_dict(s.to_dict()), s)
        with self.assertRaises(Exception):
            s.iteration = 5


if __name__ == "__main__":
    unittest.main()
