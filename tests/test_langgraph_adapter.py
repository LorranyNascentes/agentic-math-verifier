import importlib.util
import unittest

from stem_agents import AgentState, build_graph
from stem_agents.demo import PROBLEM, SCRIPT
from stem_agents.llm import ScriptedLLM

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(HAS_LANGGRAPH, "langgraph not installed (pip install '.[langgraph]')")
class LangGraphAdapterTests(unittest.TestCase):
    def test_same_workflow_same_result_on_both_engines(self):
        from stem_agents.langgraph_adapter import run_on_langgraph

        native = build_graph(ScriptedLLM(SCRIPT)).run(AgentState(problem=PROBLEM)).state
        lg = run_on_langgraph(build_graph(ScriptedLLM(SCRIPT)), AgentState(problem=PROBLEM))
        self.assertEqual(lg.status, "accepted")
        self.assertEqual(lg.iteration, native.iteration)
        self.assertEqual([r.status for r in lg.results], [r.status for r in native.results])
        self.assertEqual(lg.feedback_log, native.feedback_log)


if __name__ == "__main__":
    unittest.main()
