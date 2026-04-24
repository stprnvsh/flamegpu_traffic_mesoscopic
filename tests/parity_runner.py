from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ParityTolerance:
    flow_rel_tol: float = 0.08
    speed_rel_tol: float = 0.08
    queue_rel_tol: float = 0.12


def compare_metrics(sumo: Dict[str, float], flamegpu: Dict[str, float], tol: ParityTolerance) -> Dict[str, Any]:
    def rel_err(a: float, b: float) -> float:
        if a == 0:
            return 0.0 if b == 0 else 1.0
        return abs(a - b) / abs(a)

    flow_err = rel_err(sumo.get("flow", 0.0), flamegpu.get("flow", 0.0))
    speed_err = rel_err(sumo.get("speed", 0.0), flamegpu.get("speed", 0.0))
    queue_err = rel_err(sumo.get("queue", 0.0), flamegpu.get("queue", 0.0))
    passed = flow_err <= tol.flow_rel_tol and speed_err <= tol.speed_rel_tol and queue_err <= tol.queue_rel_tol
    return {
        "pass": passed,
        "flow_rel_err": flow_err,
        "speed_rel_err": speed_err,
        "queue_rel_err": queue_err,
    }


def run_parity_scenarios(scenarios: List[Dict[str, Dict[str, float]]], tol: ParityTolerance) -> Dict[str, Any]:
    results = []
    for s in scenarios:
        res = compare_metrics(s["sumo"], s["flamegpu"], tol)
        results.append({"name": s["name"], **res})
    return {"pass": all(r["pass"] for r in results), "scenarios": results}
