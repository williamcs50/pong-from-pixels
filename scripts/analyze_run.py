# scripts/analyze_run.py
# Print every figure the run reports in docs/runs/, from a run's own episodes.csv,
# gpu_log.csv, metrics.csv, evaluation files and run_config.json. Reads only, so it
# can be pointed at a finished run or at one still going. Sections are named and
# ordered after the run write ups, so each figure is found under the same heading.
#
#   python scripts/analyze_run.py docs/runs/run-02
#   python scripts/analyze_run.py docs/runs/run-02 --compare docs/runs/run-01

import argparse
import bisect
import csv
import json
import math
import os
import subprocess
from datetime import datetime

TRAILING = 100  # the window the bar and the pre registration are both defined over
PLATEAU = 1.0  # Run 2's pre registered plateau threshold, per 200,000 env steps
EVAL_EPISODES = 100
ASPIRATION = 18.0
BAR = 0.0  # docs/PLANS.md: a mean above 0 beats the built in opponent
RALLY_STANDARD_ERRORS = 2  # run-02.md's Clip selection rule for first rally
REPORT_EVERY = 200_000  # the evaluation sweep's --every, so tables line up with it
SHUTOUT = -21.0
POINTS_TO_WIN = 21
TEMPERATURE_LIMIT_C = 88
AGREEMENT_STANDARD_ERRORS = 2
NETWORKS_PER_UPDATE = 2  # Q on the batch's states, target on its next states

# The stretches run-02.md's Instrumentation section reports.
Q_MAX_BREAKS = (3_000_000, 5_000_000)
LATE_FROM = 3_000_000
FINAL_STRETCH = 1_000_000
FIRST_UPDATE_ROWS = 2

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
BYTES_PER_GB = 10**9
BYTES_PER_GIB = 2**30
PERCENT = 100

# Measured by Check 6 in scripts/visual_check.py.
PADDLE_GROUPS = {"up": (2, 4), "down": (3, 5), "still": (0, 1)}

# Set after the Run 2 data had been seen, so several are reported rather than one.
ONSET_FRACTIONS = (0.05, 0.10, 0.20, 0.30)

LABEL_WIDTH = 34
INDENT = "  "

CONFIG_KEYS = (
    "git_commit", "git_dirty", "started_at", "total_steps", "buffer_capacity", "warmup_steps",
    "train_every", "batch_size", "target_sync_every_updates", "epsilon_start", "epsilon_end",
    "epsilon_decay_steps", "learning_rate", "gamma", "repeat_action_probability",
    "sticky_actions_readback", "seed", "checkpoint_every_steps", "metrics_every_steps",
    "probe_states", "probe_sha256", "device_name", "n_actions",
    "python_version", "python_executable", "torch_version", "torch_cuda_version", "cudnn_version",
    "gymnasium_version", "ale-py_version", "numpy_version", "opencv-python-headless_version",
    "imageio-ffmpeg_version",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report a run's figures from its own logs.")
    parser.add_argument("run_dir", help="directory holding episodes.csv, gpu_log.csv and run_config.json")
    parser.add_argument("--compare", help="another run directory to compare this one against")
    return parser.parse_args()


def heading(name: str) -> None:
    print(f"\n== {name} ==")


def line(label: str, value: object, depth: int = 0) -> None:
    print(f"{INDENT * depth}{label:<{LABEL_WIDTH - len(INDENT) * depth - 1}} {value}")


def read_csv(path: str) -> list[dict[str, str]] | None:
    return list(csv.DictReader(open(path, newline=""))) if os.path.exists(path) else None


def load_run(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, "run_config.json")
    run = {
        "dir": run_dir,
        "config": json.load(open(config_path)) if os.path.exists(config_path) else None,
        "episodes": read_csv(os.path.join(run_dir, "episodes.csv")),
        "metrics": read_csv(os.path.join(run_dir, "metrics.csv")),
        "eval": [],
        "lengths": read_csv(os.path.join(run_dir, "eval_lengths.csv")),
    }
    # Run 1's seed 0 sweep predates the seed, std_reward and elapsed_s columns.
    for seed, name in (("0", "eval.csv"), ("1", "eval_seed1.csv")):
        rows = read_csv(os.path.join(run_dir, name))
        if rows:
            for r in rows:
                r.setdefault("seed", seed)
            run["eval"].append(rows)
    rows = run["episodes"]
    run["trail"] = trailing_means([float(r["reward"]) for r in rows], [int(r["env_step"]) for r in rows])
    run["rates"] = rates(rows)
    run["gpu"] = read_gpu(os.path.join(run_dir, "gpu_log.csv"))
    return run


def trailing_means(rewards: list[float], steps: list[int]) -> list[tuple[int, float]]:
    # (env_step, mean over the previous TRAILING episodes) once enough exist.
    return [
        (steps[i], sum(rewards[i - TRAILING + 1:i + 1]) / TRAILING)
        for i in range(TRAILING - 1, len(rewards))
    ]


def nearest(trail: list[tuple[int, float]], step: int) -> tuple[int, float]:
    return min(trail, key=lambda row: abs(row[0] - step))


def trailing_at(trail: list[tuple[int, float]], step: int) -> float | None:
    # From the last episode that ended at or before step.
    i = bisect.bisect_right([s for s, _ in trail], step) - 1
    return trail[i][1] if i >= 0 else None


def rates(rows: list[dict[str, str]]) -> dict:
    wall = float(rows[-1]["wall_clock_s"])
    # The fill phase has no updates in it and runs several times faster, so the two
    # rates are reported separately. A rate averaged across both describes neither.
    fill = [r for r in rows if int(r["gradient_updates"]) == 0]
    post = [r for r in rows if int(r["gradient_updates"]) > 0]
    result = {"overall": int(rows[-1]["env_step"]) / wall, "wall": wall, "fill": None, "post": None}
    if fill:
        result["fill"] = int(fill[-1]["env_step"]) / float(fill[-1]["wall_clock_s"])
    if len(post) > 1:
        span = int(post[-1]["env_step"]) - int(post[0]["env_step"])
        elapsed = float(post[-1]["wall_clock_s"]) - float(post[0]["wall_clock_s"])
        if elapsed > 0:
            result["post"] = span / elapsed
            result["post_span"] = (int(post[0]["env_step"]), int(post[-1]["env_step"]))
    return result


def read_gpu(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.reader(open(path)) if r][1:]
    if not rows:
        return None
    # Each field arrives with its unit attached, as in "1770 MHz".
    return {
        "time": [datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f") for r in rows],
        "temp": [int(r[1]) for r in rows],
        "clock": [int(r[2].split()[0]) for r in rows],
        "power": [float(r[3].split()[0]) for r in rows],
        "util": [int(r[4].split()[0]) for r in rows],
        "vram": [int(r[5].split()[0]) for r in rows],
    }


def standard_error(row: dict[str, str]) -> float | None:
    return float(row["std_reward"]) / math.sqrt(EVAL_EPISODES) if "std_reward" in row else None


def score(row: dict[str, str]) -> float:
    return float(row["mean_reward"])


def report_config(config: dict) -> None:
    heading("Config")
    for key in CONFIG_KEYS:
        if key in config:
            value = config[key]
            line(key, f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else value)

    commit = subprocess.run(["git", "show", "-s", "--format=%cI", config["git_commit"]],
                            capture_output=True, text=True)
    if commit.returncode == 0:
        committed = datetime.fromisoformat(commit.stdout.strip()).replace(tzinfo=None)
        lead = datetime.fromisoformat(config["started_at"]) - committed
        hours, rest = divmod(int(lead.total_seconds()), SECONDS_PER_HOUR)
        line("committed", f"{committed:%Y-%m-%d %H:%M:%S}")
        line("committed before start", f"{hours} h {rest // SECONDS_PER_MINUTE} min")


def report_results(run: dict) -> None:
    rows, trail, config = run["episodes"], run["trail"], run["config"]
    rewards = [float(row["reward"]) for row in rows]
    steps = [int(row["env_step"]) for row in rows]
    lengths = [int(row["episode_steps"]) for row in rows]
    losses = [float(row["mean_loss"]) for row in rows if row["mean_loss"]]

    heading("Results")
    line("episodes", f"{len(rows):,}")
    line("final env_step", f"{steps[-1]:,}")
    if config:
        line("short of total_steps", f"{config['total_steps'] - steps[-1]:,}")
        floor = config["epsilon_end"]
        line(f"episodes ending at {floor}", f"{sum(1 for row in rows if float(row['epsilon']) <= floor):,}")
    # From the last logged episode, not the run's total: the final episode ends
    # before the final step, so this undercounts by the updates in between.
    line("updates at last episode", f"{int(rows[-1]['gradient_updates']):,}")

    if not trail:
        line(f"trailing {TRAILING} mean", f"none, fewer than {TRAILING} episodes")
        return

    best = max(trail, key=lambda row: row[1])
    crossing = next((s for s, m in trail if m > 0), None)
    print(f"\ntrailing {TRAILING} mean")
    line("best", f"{best[1]:+.2f}", 1)
    line("best at step", f"{best[0]:,}", 1)
    line("final", f"{trail[-1][1]:+.2f}", 1)
    line("final below best", f"{best[1] - trail[-1][1]:.2f}", 1)
    line("first above 0", f"{crossing:,}" if crossing else "never", 1)

    # Plus the final step, which the interval would otherwise miss whenever the run
    # does not end exactly on a multiple.
    targets = list(range(REPORT_EVERY, steps[-1], REPORT_EVERY)) + [steps[-1]]
    if len(targets) > 1:
        print(f"\n{INDENT}     step  trailing")
        for target in targets:
            print(f"{INDENT}{target:>9,}  {nearest(trail, target)[1]:+8.2f}")

    print()
    line(f"episode length, first {TRAILING}", f"{round(sum(lengths[:TRAILING]) / TRAILING):,} steps")
    line(f"episode length, last {TRAILING}", f"{round(sum(lengths[-TRAILING:]) / TRAILING):,} steps")
    line(f"steps spanned, last {TRAILING}", f"{sum(lengths[-TRAILING:]):,}")
    line(f"shutouts, last {TRAILING}", rewards[-TRAILING:].count(SHUTOUT))
    if losses:
        line("mean loss, first", f"{losses[0]:.6f}")
        line("mean loss, last", f"{losses[-1]:.6f}")
        line("mean loss, range", f"{min(losses):.6f} to {max(losses):.6f}")


def report_evaluation(run: dict) -> None:
    sweeps, trail = run["eval"], run["trail"]
    heading("Evaluation")
    if not sweeps:
        line("evaluation files", "none")
        return

    line("seeds", len(sweeps))
    line("checkpoints", len(sweeps[0]))
    line("epsilon", ", ".join(sorted({r["epsilon"] for rows in sweeps for r in rows})))
    print(f"\n{INDENT}     step" + "".join(f"   seed {rows[0]['seed']}" for rows in sweeps)
          + "  training  seed 0 minus training")
    by_step = [{int(r["env_step"]): r for r in rows} for rows in sweeps]
    for row in sweeps[0]:
        step = int(row["env_step"])
        cells = "".join(f"  {score(s[step]):+7.2f}" if step in s else " " * 9 for s in by_step)
        if step > 0 and trail:
            t = nearest(trail, step)[1]
            cells += f"  {t:+8.2f}  {score(row) - t:+8.2f}"
        print(f"{INDENT}{step:>9,}{cells}")

    for rows in sweeps:
        best = max(rows, key=score)
        last = rows[-1]
        gap = score(best) - score(last)
        longest = max(rows, key=lambda r: int(r["mean_episode_steps"]))
        print(f"\nseed {rows[0]['seed']}")
        line("above 0", f"{sum(1 for r in rows if score(r) > 0)} of {len(rows)}", 1)
        line("best", f"{score(best):+.2f} at {int(best['env_step']):,}", 1)
        line(f"best short of +{ASPIRATION:.0f}", f"{ASPIRATION - score(best):.2f}", 1)
        line("last", f"{score(last):+.2f}", 1)
        line("best minus last", f"{gap:.2f}", 1)
        if standard_error(best) is not None:
            combined = math.hypot(standard_error(best), standard_error(last))
            line("std, first", f"{float(rows[0]['std_reward']):.2f}", 1)
            line("std, best", f"{float(best['std_reward']):.2f}", 1)
            line("std, last", f"{float(last['std_reward']):.2f}", 1)
            line("standard error, best", f"{standard_error(best):.2f}", 1)
            line("standard error, last", f"{standard_error(last):.2f}", 1)
            line("shortfall in std errors", f"{(ASPIRATION - score(best)) / standard_error(best):.1f}", 1)
            line("best minus last, combined", f"{gap / combined:.1f} standard errors", 1)
        if trail:
            line("last minus training", f"{score(last) - trail[-1][1]:+.2f}", 1)
        line("longest episodes", f"{int(longest['mean_episode_steps']):,} steps at "
             f"{int(longest['env_step']):,}, scoring {score(longest):+.2f}", 1)
        line("best checkpoint episodes", f"{int(best['mean_episode_steps']):,} steps", 1)
        if "elapsed_s" in rows[0]:
            line("sweep time", f"{sum(float(r['elapsed_s']) for r in rows) / SECONDS_PER_MINUTE:.2f} min", 1)

    if len(sweeps) > 1:
        shared = sorted(set(by_step[0]) & set(by_step[1]))
        a, b = [by_step[0][s] for s in shared], [by_step[1][s] for s in shared]
        diffs = [abs(score(x) - score(y)) for x, y in zip(a, b)]
        worst = max(range(len(diffs)), key=diffs.__getitem__)
        print(f"\nseed agreement, {len(shared)} shared checkpoints")
        line("mean absolute difference", f"{sum(diffs) / len(diffs):.2f}", 1)
        line("largest difference", f"{diffs[worst]:.2f} at {int(a[worst]['env_step']):,}", 1)
        if len(shared) < len(sweeps[0]):
            for step, diff in zip(shared, diffs):
                line(f"difference at {step:,}", f"{diff:.2f}", 1)
        if all(standard_error(rows[0]) is not None for rows in sweeps):
            beyond = sum(1 for x, y, d in zip(a, b, diffs)
                         if d > AGREEMENT_STANDARD_ERRORS * math.hypot(standard_error(x), standard_error(y)))
            line(f"beyond {AGREEMENT_STANDARD_ERRORS} std errors", beyond, 1)

    if not trail:
        return
    print(f"\nplateau windows, trailing mean moving under {PLATEAU} by endpoints or range")
    print(f"{INDENT}     from         to  endpoints  range  evaluation change")
    for x, y in zip(sweeps[0], sweeps[0][1:]):
        lo, hi = int(x["env_step"]), int(y["env_step"])
        inside = [m for s, m in trail if lo <= s <= hi]
        if not inside:
            continue
        ends = nearest(trail, hi)[1] - nearest(trail, lo)[1]
        spread = max(inside) - min(inside)
        if abs(ends) < PLATEAU or spread < PLATEAU:
            moves = ", ".join(f"seed {rows[0]['seed']} {score(s[hi]) - score(s[lo]):+.2f}"
                              for rows, s in zip(sweeps, by_step) if lo in s and hi in s)
            print(f"{INDENT}{lo:>9,}  {hi:>9,}  {ends:+9.2f}  {spread:5.2f}  {moves}")


def onset(series: list[tuple[int, float]], fraction: float) -> int | None:
    values = [v for _, v in series]
    low = values.index(min(values))
    threshold = values[low] + fraction * (max(values) - values[low])
    for i in range(low, len(values)):
        if min(values[i:]) > threshold:
            return series[i][0]
    return None


def report_instrumentation(run: dict) -> None:
    rows, config = run["metrics"], run["config"]
    heading("Instrumentation")
    if rows is None:
        line("metrics.csv", "none, this run was not instrumented")
        return

    steps = [int(r["env_step"]) for r in rows]
    updates = [int(r["gradient_updates"]) for r in rows]
    n_actions = sum(1 for k in rows[0] if k.startswith("argmax_count_"))
    counts = [[int(r[f"argmax_count_{i}"]) for i in range(n_actions)] for r in rows]
    q_max = [float(r["q_max_mean"]) for r in rows]
    q_spread = [float(r["q_spread_mean"]) for r in rows]

    print("logger checks")
    line("rows", len(rows), 1)
    if config:
        expected = list(range(0, config["total_steps"] + 1, config["metrics_every_steps"]))
        line("on schedule", f"{steps == expected}, 0 to {config['total_steps']:,} "
             f"every {config['metrics_every_steps']:,}", 1)
        line(f"argmax sums not {config['probe_states']}",
             sum(1 for c in counts if sum(c) != config["probe_states"]), 1)
    before = [i for i, u in enumerate(updates) if u == 0]
    same = len({(rows[i]["q_max_mean"], rows[i]["q_spread_mean"]) for i in before}) == 1
    line("rows before any update", f"{len(before)}, steps 0 to {steps[before[-1]]:,}", 1)
    line("their Q values", f"{'identical' if same else 'NOT identical'}, q_max {rows[0]['q_max_mean']}, "
         f"q_spread {rows[0]['q_spread_mean']}", 1)

    print("\nq_max_mean")
    lo, hi = q_max.index(min(q_max)), q_max.index(max(q_max))
    line("first", f"{q_max[0]:.4f}", 1)
    line("min", f"{q_max[lo]:.4f} at {steps[lo]:,}", 1)
    line("max", f"{q_max[hi]:.4f} at {steps[hi]:,}", 1)
    line("last", f"{q_max[-1]:.4f}", 1)
    below = next((s for s, v in zip(steps, q_max) if v < 0), None)
    above = next((steps[i] for i in range(len(q_max)) if min(q_max[i:]) > 0), None)
    line("first below 0", f"{below:,}" if below is not None else "never", 1)
    line("above 0 for good from", f"{above:,}" if above is not None else "never", 1)
    marks = [0, lo] + [steps.index(s) for s in Q_MAX_BREAKS if s in steps and steps.index(s) > lo]
    marks.append(len(steps) - 1)
    for a, b in zip(marks, marks[1:]):
        line(f"change {steps[a]:,} to {steps[b]:,}", f"{q_max[b] - q_max[a]:+.4f}", 1)
    final = [v for s, v in zip(steps, q_max) if s >= steps[-1] - FINAL_STRETCH]
    line(f"range, last {FINAL_STRETCH:,}", f"{min(final):.4f} to {max(final):.4f}", 1)

    print("\nq_spread_mean")
    lo, hi = q_spread.index(min(q_spread)), q_spread.index(max(q_spread))
    line("first", f"{q_spread[0]:.4f}", 1)
    for i in [i for i, u in enumerate(updates) if u > 0][:FIRST_UPDATE_ROWS]:
        line(f"at {steps[i]:,}", f"{q_spread[i]:.4f}, after {updates[i]:,} "
             f"update{'' if updates[i] == 1 else 's'}", 1)
    line("min", f"{q_spread[lo]:.4f} at {steps[lo]:,}", 1)
    line("max", f"{q_spread[hi]:.4f} at {steps[hi]:,}", 1)
    line("last", f"{q_spread[-1]:.4f}", 1)
    late = [v for s, v in zip(steps, q_spread) if s >= LATE_FROM]
    if late:
        line(f"range from {LATE_FROM:,}", f"{min(late):.4f} to {max(late):.4f}", 1)
        line(f"mean from {LATE_FROM:,}", f"{sum(late) / len(late):.4f}", 1)

    trail = run["trail"]
    spread = list(zip(steps, q_spread))
    reward = [(s, trailing_at(trail, s)) for s, _ in spread if trailing_at(trail, s) is not None]
    if reward:
        print("\nonset, first step above minimum plus fraction of range, held to the end")
        print(f"{INDENT}fraction    q_spread      reward        lead")
        for fraction in ONSET_FRACTIONS:
            a, b = onset(spread, fraction), onset(reward, fraction)
            show = lambda x: "none" if x is None else f"{x:,}"
            lead = show(b - a) if a is not None and b is not None else ""
            print(f"{INDENT}{fraction:8.2f}  {show(a):>10}  {show(b):>10}  {lead:>10}")

    if config:
        epsilon = [float(r["epsilon"]) for r in rows]
        start, end, decay = config["epsilon_start"], config["epsilon_end"], config["epsilon_decay_steps"]
        # A row at env_step N holds the epsilon computed for loop index N - 1.
        scheduled = [start + min((s - 1) / decay, 1.0) * (end - start) for s in steps[1:]]
        first_floor = next((s for s, e in zip(steps, epsilon) if e <= end), None)
        after = [e for s, e in zip(steps, epsilon) if s > decay]
        print("\nepsilon")
        line("max deviation from schedule", f"{max(abs(e - x) for e, x in zip(epsilon[1:], scheduled)):.8f}", 1)
        if decay in steps:
            line(f"row at {decay:,}", f"{epsilon[steps.index(decay)]:.6f}", 1)
        line(f"first row at {end}", f"{first_floor:,}" if first_floor else "never", 1)
        line(f"rows after {decay:,} at {end}", f"{sum(1 for e in after if e == end)} of {len(after)}", 1)

    report_paddle(steps, counts)

    if config:
        report_replay(rows, steps, run["episodes"], config)

    every = REPORT_EVERY // (steps[1] - steps[0])
    print(f"\nevery {REPORT_EVERY:,} steps: Q, paddle direction, replay")
    print(f"{INDENT}     step    q_max  q_spread   up down still  reward_frac  done_frac")
    for i in range(0, len(rows), every):
        r, c = rows[i], counts[i]
        groups = [sum(c[a] for a in PADDLE_GROUPS[g]) for g in ("up", "down", "still")]
        fractions = (f"  {float(r['batch_reward_fraction']):.5f}      {float(r['batch_done_fraction']):.6f}"
                     if r["batch_reward_fraction"] else "")
        print(f"{INDENT}{steps[i]:>9,}  {q_max[i]:>7.4f}  {q_spread[i]:>8.4f}  "
              f"{groups[0]:>3}  {groups[1]:>3}  {groups[2]:>4}{fractions}")


def report_paddle(steps: list[int], counts: list[list[int]]) -> None:
    groups = {g: [sum(c[a] for a in actions) for c in counts] for g, actions in PADDLE_GROUPS.items()}
    # A network with no preference splits the probe states evenly across the groups.
    indifferent = sum(counts[0]) / len(PADDLE_GROUPS)
    late = [i for i, s in enumerate(steps) if s >= LATE_FROM]

    print(f"\npaddle direction, against {indifferent:.0f} states per group with no preference")
    for name, values in groups.items():
        if late:
            line(f"{name}, mean from {LATE_FROM:,}", f"{sum(values[i] for i in late) / len(late):.1f}", 1)
    still = groups["still"]
    below = next((steps[i] for i in range(len(still)) if max(still[i:]) < indifferent), None)
    line("still below no preference for good from", f"{below:,}" if below is not None else "never", 1)
    above = [steps[i] for i in range(1, len(still)) if still[i] >= indifferent]
    line("rows since step 0 at or above it", ", ".join(f"{s:,}" for s in above) or "none", 1)


def report_replay(rows: list[dict[str, str]], steps: list[int], episodes: list[dict[str, str]],
                  config: dict) -> None:
    # The row at warmup_steps holds a single batch, so only full windows are summarised.
    full = config["metrics_every_steps"] // config["train_every"]
    sampled = [i for i, r in enumerate(rows) if r["batches_in_window"] and int(r["batches_in_window"]) == full]
    if not sampled:
        return
    columns = {"reward": "batch_reward_fraction", "done": "batch_done_fraction"}
    for column in columns.values():
        values = [float(rows[i][column]) for i in sampled]
        lo, hi = values.index(min(values)), values.index(max(values))
        print(f"\n{column}, {len(sampled)} full windows of {full:,} batches")
        line("first", f"{values[0]:.6f} at {steps[sampled[0]]:,}", 1)
        line("min", f"{values[lo]:.6f} at {steps[sampled[lo]]:,}", 1)
        line("max", f"{values[hi]:.6f} at {steps[sampled[hi]]:,}", 1)
        line("last", f"{values[-1]:.6f}", 1)

    # One side always reaches 21, so an episode holds 2 * 21 - |reward| points. The
    # buffer holds the last buffer_capacity steps, so the episodes ending inside that
    # span predict what a uniform sample from it should contain.
    ends = [int(e["env_step"]) for e in episodes]
    lengths = [int(e["episode_steps"]) for e in episodes]
    points = [2 * POINTS_TO_WIN - abs(float(e["reward"])) for e in episodes]
    z = {column: [] for column in columns.values()}
    for i in sampled:
        lo = bisect.bisect_right(ends, steps[i] - config["buffer_capacity"])
        hi = bisect.bisect_right(ends, steps[i])
        span = sum(lengths[lo:hi])
        if span == 0:
            continue
        n = int(rows[i]["batches_in_window"]) * config["batch_size"]
        for column, predicted in (("batch_reward_fraction", sum(points[lo:hi]) / span),
                                  ("batch_done_fraction", (hi - lo) / span)):
            z[column].append((float(rows[i][column]) - predicted) / math.sqrt(predicted * (1 - predicted) / n))

    print("\nlogged minus predicted from episodes.csv, in sampling standard errors")
    for name, column in columns.items():
        values = z[column]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        line(f"{name}, mean", f"{mean:+.2f}", 1)
        line(f"{name}, sd", f"{sd:.2f}", 1)
        line(f"{name}, within {AGREEMENT_STANDARD_ERRORS}",
             f"{sum(1 for v in values if abs(v) <= AGREEMENT_STANDARD_ERRORS)} of {len(values)}", 1)


def report_throughput(run: dict) -> None:
    r, gpu, config = run["rates"], run["gpu"], run["config"]
    heading("Throughput and hardware")
    line("wall clock", f"{r['wall'] / SECONDS_PER_HOUR:.2f} h")
    line("overall rate", f"{r['overall']:.1f} steps/s")
    if r["fill"]:
        line("fill phase rate", f"{r['fill']:.1f} steps/s")
    if r["post"]:
        line("post warmup rate", f"{r['post']:.1f} steps/s, step {r['post_span'][0]:,} "
             f"to {r['post_span'][1]:,}")

    # Absent whenever the nvidia-smi logger was not started, which is normal for a
    # warmup measuring throughput rather than temperature.
    if gpu is None:
        line("gpu_log.csv", "none, the logger was not running for this run")
    else:
        time = gpu["time"]
        gaps = [(b - a).total_seconds() for a, b in zip(time, time[1:])]
        line("gpu samples", len(time))
        line("first and last sample", f"{time[0]:%H:%M:%S} to {time[-1]:%H:%M:%S}")
        line("samples span", f"{(time[-1] - time[0]).total_seconds() / SECONDS_PER_HOUR:.2f} h")
        if gaps:
            line("sample gaps", f"{min(gaps):.3f} to {max(gaps):.3f} s")
        if config:
            started = datetime.fromisoformat(config["started_at"])
            line("logger started before run", f"{(started - time[0]).total_seconds():.0f} s")
        line("temperature", f"{min(gpu['temp'])} to {max(gpu['temp'])} C")
        line(f"readings above {TEMPERATURE_LIMIT_C} C", sum(1 for t in gpu["temp"] if t > TEMPERATURE_LIMIT_C))
        line("sm clock", f"{min(gpu['clock']):,} to {max(gpu['clock']):,} MHz")
        line("power draw", f"{min(gpu['power'])} to {max(gpu['power'])} W")
        line("utilization", f"{min(gpu['util'])} to {max(gpu['util'])} %")
        line("vram used", f"{min(gpu['vram']):,} to {max(gpu['vram']):,} MiB")

    if config and run["metrics"]:
        updates = int(run["metrics"][-1]["gradient_updates"])
        line("probe forward passes", f"{len(run['metrics']) * config['probe_states']:,}")
        line("training forward passes", f"{updates * config['batch_size'] * NETWORKS_PER_UPDATE:,}")

    if config:
        checkpoints = os.path.join(config["output_dir"], "checkpoints")
        if os.path.isdir(checkpoints):
            files = [os.path.join(checkpoints, f) for f in os.listdir(checkpoints) if f.endswith(".pt")]
            size = sum(os.path.getsize(f) for f in files)
            line("checkpoints on disk", f"{len(files)} files, {size / BYTES_PER_GIB:.2f} GiB, "
                 f"{size / BYTES_PER_GB:.2f} GB")


def report_clip_selection(run: dict) -> None:
    sweeps, rescore = run["eval"], run["lengths"]
    heading("Clip selection")
    if not sweeps:
        line("evaluation files", "none")
        return
    seed0 = {int(r["env_step"]): r for r in sweeps[0]}
    steps = sorted(seed0)

    line("random", f"{steps[0]:,}")

    if rescore is None:
        line("first rally", "no eval_lengths.csv, so the rule cannot be applied")
    else:
        rows = sorted(rescore, key=lambda r: int(r["env_step"]))
        matching = sum(1 for r in rows if int(r["env_step"]) in seed0 and all(
            r[k] == seed0[int(r["env_step"])][k] for k in ("mean_reward", "std_reward", "mean_episode_steps")))
        print("\nrescore with episode length spread")
        line("checkpoints", f"{rows[0]['env_step']} to {int(rows[-1]['env_step']):,}", 1)
        line("matching eval.csv", f"{matching} of {len(rows)}", 1)
        line("rescore time", f"{sum(float(r['elapsed_s']) for r in rows) / SECONDS_PER_MINUTE:.1f} min", 1)

        mean = lambda r: float(r["mean_episode_steps"])
        error = lambda r: float(r["std_episode_steps"]) / math.sqrt(EVAL_EPISODES)
        base = rows[0]
        print(f"\nfirst rally, mean episode length beating step {int(base['env_step']):,}'s "
              f"by more than {RALLY_STANDARD_ERRORS} combined standard errors")
        print(f"{INDENT}     step  mean steps  beats by     bar  fires")
        print(f"{INDENT}{int(base['env_step']):>9,}  {mean(base):>10,.0f}  base")
        rally = None
        for r in rows[1:]:
            beats = mean(r) - mean(base)
            bar = RALLY_STANDARD_ERRORS * math.hypot(error(base), error(r))
            fires = beats > bar
            print(f"{INDENT}{int(r['env_step']):>9,}  {mean(r):>10,.0f}  {beats:>+8,.0f}  {bar:>6.1f}  "
                  f"{'yes' if fires else 'no'}")
            if fires:
                rally = int(r["env_step"])
                break
        line("first rally", f"{rally:,}" if rally is not None
             else f"not fired by {int(rows[-1]['env_step']):,}, rescore further")

    over = next((s for s in steps if score(seed0[s]) > BAR), None)
    best = max(steps, key=lambda s: score(seed0[s]))
    print()
    line("first over the bar", f"{over:,}, seed 0 {score(seed0[over]):+.2f}" if over is not None else "never")
    seed1 = {int(r["env_step"]): r for r in sweeps[1]} if len(sweeps) > 1 else {}
    quoted = f", seed 1 {score(seed1[best]):+.2f}" if best in seed1 else ""
    line("trained", f"{best:,}, seed 0 {score(seed0[best]):+.2f}{quoted}")
    if best == over:
        line("note", "trained and first over the bar are one checkpoint, so one clip covers both")


def report_compare(run: dict, other: dict) -> None:
    heading(f"Compared with {other['dir']}")

    strip = lambda row: {k: v for k, v in row.items() if k != "wall_clock_s"}
    pairs = list(zip(run["episodes"], other["episodes"]))
    first = next((i for i, (a, b) in enumerate(pairs) if strip(a) != strip(b)), None)
    if first is None:
        line("identical episodes", f"all {len(pairs):,} shared, apart from wall clock")
    else:
        prior = int(run["episodes"][first - 1]["env_step"]) if first else 0
        line("identical episodes", f"{first}, apart from wall clock")
        line("first differing episode", f"{first + 1}, steps {prior:,} to "
             f"{int(run['episodes'][first]['env_step']):,}")

    if run["trail"] and other["trail"]:
        at = other["trail"][-1][0]
        mine, theirs = nearest(run["trail"], at)[1], other["trail"][-1][1]
        print(f"\ntrailing mean at {at:,}")
        line("this", f"{mine:+.2f}", 1)
        line("other", f"{theirs:+.2f}", 1)
        line("gap", f"{theirs - mine:.2f}", 1)

    for key in ("fill", "post"):
        mine, theirs = run["rates"][key], other["rates"][key]
        if mine and theirs:
            print(f"\n{key} rate")
            line("this", f"{mine:.1f} steps/s", 1)
            line("other", f"{theirs:.1f} steps/s", 1)
            line("this slower by", f"{(theirs - mine) / theirs * PERCENT:.1f}%", 1)
            line("other faster by", f"{(theirs / mine - 1) * PERCENT:.1f}%", 1)

    config = run["config"]
    if config and other["rates"]["fill"] and other["rates"]["post"]:
        warmup, total = config["warmup_steps"], config["total_steps"]
        projected = (warmup / other["rates"]["fill"] + (total - warmup) / other["rates"]["post"]) / SECONDS_PER_HOUR
        actual = run["rates"]["wall"] / SECONDS_PER_HOUR
        print(f"\nprojection from the other run's rates")
        line(f"projected, {total:,} steps", f"{projected:.2f} h", 1)
        line("actual", f"{actual:.2f} h", 1)
        line("actual over projection", f"{(actual / projected - 1) * PERCENT:.1f}%", 1)

    if run["gpu"] and other["gpu"]:
        print("\npeaks")
        line("sm clock, this", f"{max(run['gpu']['clock']):,} MHz", 1)
        line("sm clock, other", f"{max(other['gpu']['clock']):,} MHz", 1)
        line("power, this", f"{max(run['gpu']['power'])} W", 1)
        line("power, other", f"{max(other['gpu']['power'])} W", 1)

    if run["eval"] and other["eval"] and run["trail"] and other["trail"]:
        print("\nfinal evaluation against training")
        for name, r in (("this", run), ("other", other)):
            last = score(r["eval"][0][-1])
            line(f"{name}, evaluation", f"{last:+.2f}", 1)
            line(f"{name}, training", f"{r['trail'][-1][1]:+.2f}", 1)
            line(f"{name}, gap", f"{last - r['trail'][-1][1]:+.2f}", 1)
        line("evaluation, this minus other", f"{score(run['eval'][0][-1]) - score(other['eval'][0][-1]):+.2f}", 1)


if __name__ == "__main__":
    args = parse_args()
    run = load_run(args.run_dir)

    if run["config"]:
        report_config(run["config"])
    report_results(run)
    report_evaluation(run)
    report_instrumentation(run)
    report_throughput(run)
    report_clip_selection(run)
    if args.compare:
        report_compare(run, load_run(args.compare))
