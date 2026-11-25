import json
import unittest

from stem_agents.llm import ScriptedLLM
from stem_agents.models import Plan
from stem_agents.structured import StructuredOutputError, extract_json, generate


class StructuredTests(unittest.TestCase):
    def test_extracts_json_from_fences_and_prose(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nDone.'), {"a": [1, 2]})

    def test_repairs_after_invalid_output(self):
        llm = ScriptedLLM({"planner": ['{"steps": "not a list"}', json.dumps({"steps": ["ok"]})]})
        plan = generate(llm, "planner", Plan, "sys", "user")
        self.assertEqual(plan.steps, ["ok"])
        self.assertIn("previous reply was invalid", llm.calls[1][1])  # error fed back to the model

    def test_gives_up_after_retries(self):
        with self.assertRaises(StructuredOutputError):
            generate(ScriptedLLM({"planner": ["garbage"]}), "planner", Plan, "s", "u", retries=1)


if __name__ == "__main__":
    unittest.main()
