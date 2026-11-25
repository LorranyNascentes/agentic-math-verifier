import unittest

from stem_agents.models import Claim, Variable
from stem_agents.tools import UnsafeExpression, check_identity, check_monte_carlo, default_registry, safe_eval, _namespace

X = [Variable(name="x", low=0.1, high=2.0)]


def claim(lhs, rhs, **kw):
    return Claim(id="c", description="", kind="identity", lhs=lhs, rhs=rhs, variables=kw.pop("variables", X), **kw)


class ToolTests(unittest.TestCase):
    def test_symbolic_proof(self):
        self.assertEqual(check_identity(claim("sin(x)**2 + cos(x)**2", "1")).status, "verified_symbolic")

    def test_refutation_returns_counterexample(self):
        r = check_identity(claim("x**2", "x"))
        self.assertEqual(r.status, "refuted")
        self.assertIn("x", r.counterexample)

    def test_numeric_fallback_on_large_expression(self):
        d = {"u": "exp(x) + exp(2*x) + exp(3*x) + exp(4*x) + exp(5*x) + exp(6*x)"}
        r = check_identity(claim("diff(u, x)", "exp(x) + 2*exp(2*x) + 3*exp(3*x) + 4*exp(4*x) + 5*exp(5*x) + 6*exp(6*x)",
                                 definitions=d))
        self.assertTrue(r.verified)

    def test_sandbox_rejects_code_execution(self):
        ns = _namespace({})
        for bad in ["__import__('os').system('echo hi')", "(1).__class__", "[x for x in range(3)]",
                    "open('/etc/passwd')", "lambda: 1"]:
            with self.assertRaises(UnsafeExpression, msg=bad):
                safe_eval(bad, ns)

    def test_registry_converts_exceptions_to_errors(self):
        res = default_registry().call("check_identity", claim=claim("__import__('os')", "0"))
        self.assertFalse(res.ok)
        self.assertIn("UnsafeExpression", res.error)

    def test_monte_carlo_flags_wrong_price(self):
        p = {"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "t": 0.0, "T": 1.0}
        defs = {"tau": "T - t", "d1": "(log(S/K)+(r+sigma**2/2)*tau)/(sigma*sqrt(tau))",
                "d2": "d1 - sigma*sqrt(tau)", "C": "S*Phi(d1) - K*exp(-r*tau)*Phi(d2)", "bad": "C + 1"}
        good = Claim(id="g", description="", kind="monte_carlo", lhs="C", definitions=defs, params=p)
        bad = Claim(id="b", description="", kind="monte_carlo", lhs="bad", definitions=defs, params=p)
        self.assertTrue(check_monte_carlo(good).verified)
        self.assertEqual(check_monte_carlo(bad).status, "refuted")


if __name__ == "__main__":
    unittest.main()
