from src.core.simulation import evaluate_kpi_gate
from tests.parity_runner import compare_metrics, ParityTolerance, run_parity_scenarios


if __name__ == "__main__":
    baseline = {"sim_seconds_per_wall_second": 100.0}
    candidate = {"sim_seconds_per_wall_second": 98.0}
    print("KPI gate:", evaluate_kpi_gate(baseline, candidate))

    sumo_metrics = {"flow": 1200.0, "speed": 13.5, "queue": 40.0}
    flamegpu_metrics = {"flow": 1160.0, "speed": 13.0, "queue": 43.0}
    print("Parity check:", compare_metrics(sumo_metrics, flamegpu_metrics, ParityTolerance()))

    scenarios = [
        {"name": "heavy_jam", "sumo": {"flow": 800.0, "speed": 6.0, "queue": 120.0}, "flamegpu": {"flow": 770.0, "speed": 5.8, "queue": 128.0}},
        {"name": "disconnected_network", "sumo": {"flow": 500.0, "speed": 9.5, "queue": 30.0}, "flamegpu": {"flow": 485.0, "speed": 9.2, "queue": 31.0}},
        {"name": "rerouter_active", "sumo": {"flow": 1100.0, "speed": 12.0, "queue": 60.0}, "flamegpu": {"flow": 1060.0, "speed": 11.6, "queue": 64.0}},
    ]
    print("Scenario parity:", run_parity_scenarios(scenarios, ParityTolerance()))
