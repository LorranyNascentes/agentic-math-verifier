import unittest

from stem_agents import AgentState, build_graph
from stem_agents.demo import PROBLEM, SCRIPT
from stem_agents.llm import ScriptedLLM


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.llm = ScriptedLLM(SCRIPT)
        self.result = build_graph(self.llm).run(AgentState(problem=PROBLEM))

    def test_refute_then_repair_loop(self):
        s = self.result.state
        self.assertEqual(s.status, "accepted")
        self.assertEqual(s.iteration, 2)
        nodes = [e.node for e in self.result.trace]
        self.assertEqual(nodes, ["planner", "hypothesizer", "formalizer", "validator", "critic",
                                 "hypothesizer", "formalizer", "validator", "critic", "reporter"])

    def test_all_claims_verified_with_evidence(self):
        s = self.result.state
        self.assertEqual({r.claim_id for r in s.results}, {"bs_pde", "delta", "put_call_parity", "monte_carlo"})
        self.assertTrue(all(r.verified for r in s.results), [r for r in s.results if not r.verified])

    def test_first_attempt_was_caught_by_the_tools_not_the_llm(self):
        first_critic_prompt = [u for role, u in self.llm.calls if role == "critic"][0]
        self.assertIn('"status": "refuted"', first_critic_prompt)
        self.assertIn("counterexample", first_critic_prompt)

    def test_feedback_reaches_next_iteration(self):
        second_hyp_prompt = [u for role, u in self.llm.calls if role == "hypothesizer"][1]
        self.assertIn("wrong sign", second_hyp_prompt)

    def test_exhaustion_when_never_fixed(self):
        script = dict(SCRIPT, formalizer=[SCRIPT["formalizer"][0]])  # always the flawed formalization
        s = build_graph(ScriptedLLM(script)).run(AgentState(problem=PROBLEM, max_iterations=2)).state
        self.assertEqual((s.status, s.iteration), ("exhausted", 2))
        self.assertIn("exhausted", s.report_md)

    def test_report_contains_evidence_table(self):
        self.assertIn("| bs_pde | verified_", self.result.state.report_md)


if __name__ == "__main__":
    unittest.main()
