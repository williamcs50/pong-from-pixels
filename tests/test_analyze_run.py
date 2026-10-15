import importlib.util
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "analyze_run.py")
RUNS = os.path.join(REPO_ROOT, "docs", "runs")

spec = importlib.util.spec_from_file_location("analyze_run", SCRIPT)
analyze_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_run)

# Exact phrases quoting figures no run's files contain: facts about the hardware or the
# game, pre registration constants, software versions, and arithmetic on printed figures.
EXTERNAL = (
    "1e-4", "+1 and -1 per point", "21 to 3", "21 to 19", "21 to 0 or 21 to 2", "of 6,144",
    "rated around 160", "13.5 MB", "Oct 10", "Python 3.11.0", "torch 2.12.1+cu126", "CUDA 12.6",
    "gymnasium 1.3.0", "ale-py 0.12.0", "numpy 2.4.6", "before step 2,300,000",
    "2.02, 2.27 and 2.56", "missing by 415,875 steps", "415,875 steps late", "step 250,001",
    "the 90 checkpoints", "12 of the 30 windows", "7 to 9 points", "about 95 percent",
    "roughly 95 million", "8,000,000 frames", "Nature's 50,000,000", "Nature's 1,000,000",
    "1.2M", "across 200 episodes", "gained 2.56 points", "about 5 combined", "about 50 minutes",
    "51.3 W", "or 51 W",
    "first 30 seconds", "450 frames at 15 fps", "frame 450", "six Q values", "304 steps longer", "last 30 seconds", "last 30 end", "last 450 frames",
    "seeds 0 to 8", "six of the nine", "seeds 0, 1, 2", "at least 3 returns within 30 seconds",
    "starts 2 seconds before",
)


def run(*args: str) -> str:
    """Run analyze_run.py with these arguments and return what it printed."""
    result = subprocess.run([sys.executable, SCRIPT, *args], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def numbers(text: str) -> set[str]:
    """Return every number in text, without commas or a leading plus sign."""
    return {n.replace(",", "").lstrip("+").rstrip(".") for n in re.findall(r"[-+]?\d[\d,]*\.?\d*", text)}


def unprinted(doc: str, output: str) -> list[str]:
    """Return the numbers a write up quotes that the analyzer output never prints."""
    # Commands, code, links, dates and clock times are not figures.
    body = re.sub(r"```.*?```", "", doc, flags=re.S)
    body = re.sub(r"`[^`]*`|\[[^\]]*\]\([^)]*\)|20\d\d-\d\d-\d\d|\d\d:\d\d(:\d\d)?", "", body)
    body = re.sub(r"\s+", " ", body)
    for phrase in EXTERNAL:
        body = body.replace(phrase, "")
    # Prose says "4.30 below" where the table prints -4.30, so the sign is not compared.
    printed = {n.lstrip("-") for n in numbers(output)}
    return sorted(n for n in numbers(body) if n.lstrip("-") not in printed)


def outputs_for(name: str) -> str:
    """Return the analyzer output for a write up and every run directory it draws on."""
    runs = {
        "run-02": [("run-02",), ("run-02", "--compare", "run-01"), ("run-02", "--compare", "run-02-warmup"),
                   ("run-01",), ("run-00-warmup",)],
        "run-01": [("run-01",), ("run-00-warmup",)],
    }[name]
    return "\n".join(run(*[os.path.join(RUNS, a) if not a.startswith("--") else a for a in args])
                     for args in runs)


def test_every_run_directory_runs() -> None:
    for name in ("run-00-warmup", "run-01", "run-02-warmup", "run-02"):
        run(os.path.join(RUNS, name))
    print("PASS  analyze_run.py completes on every run directory")


def test_run_02_md_figures_are_printed() -> None:
    missing = unprinted(open(os.path.join(RUNS, "run-02.md"), encoding="utf-8").read(), outputs_for("run-02"))
    assert not missing, f"run-02.md quotes figures analyze_run.py does not print: {missing}"
    print("PASS  every figure in run-02.md is printed by analyze_run.py or listed as external")


def test_run_01_md_figures_are_printed() -> None:
    missing = unprinted(open(os.path.join(RUNS, "run-01.md"), encoding="utf-8").read(), outputs_for("run-01"))
    assert not missing, f"run-01.md quotes figures analyze_run.py does not print: {missing}"
    print("PASS  every figure in run-01.md is printed by analyze_run.py or listed as external")


def test_figure_check_can_fail() -> None:
    assert unprinted("The best checkpoint scored +16.97.", "best +16.97") == []
    assert unprinted("The best checkpoint scored +16.98.", "best +16.97") == ["16.98"]
    print("PASS  the figure check flags a number the output does not contain")


def test_headline_figures() -> None:
    output = run(os.path.join(RUNS, "run-02"))
    for expected in ("first above 0                   2,715,875",
                     "first rally                       600,000",
                     "first over the bar                2,800,000",
                     "trained                           5,400,000",
                     "matching eval.csv               6 of 6"):
        assert expected in output, f"missing from analyze_run.py output: {expected!r}"
    print("PASS  the crossing and all four clip picks print as run-02.md states them")


def test_trailing_means_known_values() -> None:
    original = analyze_run.TRAILING
    analyze_run.TRAILING = 3
    try:
        trail = analyze_run.trailing_means([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    finally:
        analyze_run.TRAILING = original
    assert trail == [(30, 2.0), (40, 3.0), (50, 4.0)], f"trailing means {trail}"
    print("PASS  trailing_means matches hand worked windows of 3")


def test_trailing_at_uses_last_episode_ended() -> None:
    trail = [(30, 2.0), (40, 3.0), (50, 4.0)]
    assert analyze_run.trailing_at(trail, 29) is None
    assert analyze_run.trailing_at(trail, 30) == 2.0
    assert analyze_run.trailing_at(trail, 49) == 3.0
    print("PASS  trailing_at reads the last episode that ended by each step")


def test_onset_known_series() -> None:
    assert analyze_run.onset([(0, 0), (1, 5), (2, 10), (3, 0.1), (4, 10)], 0.05) == 4
    assert analyze_run.onset([(0, 0), (1, 10), (2, 0)], 0.05) is None
    assert analyze_run.onset([(0, 3), (1, 0), (2, 1), (3, 2)], 0.5) == 3
    print("PASS  onset resets on a dip, returns None for a fall back, and ignores a maximum before the minimum")


def test_rates_split_fill_and_training() -> None:
    rows = [{"env_step": "100", "gradient_updates": "0", "wall_clock_s": "1.0"},
            {"env_step": "200", "gradient_updates": "0", "wall_clock_s": "2.0"},
            {"env_step": "300", "gradient_updates": "5", "wall_clock_s": "4.0"},
            {"env_step": "500", "gradient_updates": "55", "wall_clock_s": "8.0"}]
    rates = analyze_run.rates(rows)
    assert rates["fill"] == 100.0, f"fill {rates['fill']}"
    assert rates["post"] == 50.0, f"post {rates['post']}"
    assert rates["overall"] == 62.5, f"overall {rates['overall']}"
    print("PASS  rates separates the fill phase from training")


if __name__ == "__main__":
    test_every_run_directory_runs()
    test_run_02_md_figures_are_printed()
    test_run_01_md_figures_are_printed()
    test_figure_check_can_fail()
    test_headline_figures()
    test_trailing_means_known_values()
    test_trailing_at_uses_last_episode_ended()
    test_onset_known_series()
    test_rates_split_fill_and_training()
    print("\nAll tests passed!")
