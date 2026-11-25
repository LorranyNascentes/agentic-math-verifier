import tempfile
import unittest
from dataclasses import dataclass

from stem_agents.graph import END, GraphError, StateGraph


@dataclass(frozen=True)
class S:
    n: int = 0
    def to_dict(self): return {"n": self.n}
    @classmethod
    def from_dict(cls, d): return cls(**d)


def inc(s): return {"n": s.n + 1}


class GraphTests(unittest.TestCase):
    def loop(self):
        return (StateGraph().add_node("a", inc).add_node("b", inc).set_entry_point("a")
                .add_edge("a", "b").add_conditional_edges("b", lambda s: "a" if s.n < 6 else END, ["a", END]))

    def test_conditional_loop_terminates(self):
        res = self.loop().run(S())
        self.assertEqual(res.state.n, 6)
        self.assertEqual([e.node for e in res.trace], list("ababab"))

    def test_max_steps_guards_infinite_loops(self):
        g = StateGraph().add_node("a", inc).set_entry_point("a").add_edge("a", "a")
        with self.assertRaisesRegex(GraphError, "max_steps"):
            g.run(S(), max_steps=5)

    def test_bad_router_target_is_rejected(self):
        g = StateGraph().add_node("a", inc).set_entry_point("a").add_conditional_edges("a", lambda s: "zzz", ["a"])
        with self.assertRaises(GraphError):
            g.run(S())

    def test_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            g = self.loop()
            full = g.run(S(), checkpoint_dir=d)
            resumed = g.resume(f"{d}/03_a.json", S)  # crash-recovery from the middle of the run
            self.assertEqual(resumed.state, full.state)
            self.assertEqual([e.node for e in resumed.trace], list("bab"))


if __name__ == "__main__":
    unittest.main()
