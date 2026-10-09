# scripts/analyze_run.py
# Print every figure the run reports in docs/runs/, from a run's own episodes.csv,
# gpu_log.csv and run_config.json. Reads only, so it can be pointed at a finished
# run or at one still going.
#
#   python scripts/analyze_run.py docs/runs/run-01

import argparse
import csv
import json
import os

TRAILING = 100  # the window the bar and the pre registration are both defined over


def parse_args():
    parser = argparse.ArgumentParser(description="Report a run's figures from its own logs.")
    parser.add_argument("run_dir", help="directory holding episodes.csv, gpu_log.csv and run_config.json")
    return parser.parse_args()


def trailing_means(rewards, steps):
    # (env_step, mean over the previous TRAILING episodes) once enough exist.
    return [
        (steps[i], sum(rewards[i - TRAILING + 1:i + 1]) / TRAILING)
        for i in range(TRAILING - 1, len(rewards))
    ]


def report_episodes(path):
    rows = list(csv.DictReader(open(path)))
    rewards = [float(r["reward"]) for r in rows]
    steps = [int(r["env_step"]) for r in rows]
    lengths = [int(r["episode_steps"]) for r in rows]
    losses = [float(r["mean_loss"]) for r in rows if r["mean_loss"]]
    wall = float(rows[-1]["wall_clock_s"])

    print(f"episodes          {len(rows):,}")
    print(f"final env_step    {steps[-1]:,}")
    # From the last logged episode, not the run's total: the final episode ends
    # before the final step, so this undercounts by the updates in between.
    print(f"updates at last ep {int(rows[-1]['gradient_updates']):,}")
    print(f"wall clock        {wall / 3600:.2f} h")
    print(f"overall rate      {steps[-1] / wall:.1f} steps/s")

    # The fill phase has no updates in it and runs several times faster, so the two
    # rates are reported separately. A rate averaged across both describes neither.
    post = [r for r in rows if int(r["gradient_updates"]) > 0]
    if post:
        first, last = post[0], post[-1]
        span = int(last["env_step"]) - int(first["env_step"])
        elapsed = float(last["wall_clock_s"]) - float(first["wall_clock_s"])
        if elapsed > 0:
            print(f"post warmup rate  {span / elapsed:.1f} steps/s "
                  f"(step {int(first['env_step']):,} to {int(last['env_step']):,})")

    if len(rows) < TRAILING:
        print(f"\nfewer than {TRAILING} episodes, no trailing mean")
        return

    trail = trailing_means(rewards, steps)
    crossing = next((s for s, m in trail if m > 0), None)

    print(f"\ntrailing {TRAILING} mean")
    print(f"  best            {max(m for _, m in trail):+.2f}")
    print(f"  final           {trail[-1][1]:+.2f}")
    print(f"  first above 0   {f'step {crossing:,}' if crossing else 'never'}")

    # Every 200,000 plus the final step, which the interval would otherwise miss
    # whenever the run does not end exactly on a multiple.
    targets = list(range(200_000, steps[-1], 200_000)) + [steps[-1]]

    if len(targets) > 1:
        print("\n  by step")
        for target in targets:
            nearest = min(trail, key=lambda row: abs(row[0] - target))
            print(f"    {target:>9,}  {nearest[1]:+6.2f}")

    print(f"\nepisode length    first {TRAILING}: {round(sum(lengths[:TRAILING]) / TRAILING):,} steps, "
          f"last {TRAILING}: {round(sum(lengths[-TRAILING:]) / TRAILING):,} steps")
    print(f"shutouts          {rewards[-TRAILING:].count(-21.0)} in the last {TRAILING} episodes")

    if losses:
        print(f"mean loss         first {losses[0]:.6f}, last {losses[-1]:.6f}, "
              f"range {min(losses):.6f} to {max(losses):.6f}")


def report_gpu(path):
    # Absent whenever the nvidia-smi logger was not started, which is normal for a
    # warmup measuring throughput rather than temperature.
    if not os.path.exists(path):
        print()
        print("no gpu_log.csv, the logger was not running for this run")
        return

    rows = [r for r in csv.reader(open(path)) if r][1:]

    if not rows:
        print("\ngpu_log.csv has no samples")
        return

    # Each field arrives with its unit attached, as in "1770 MHz".
    temp = [int(r[1]) for r in rows]
    clock = [int(r[2].split()[0]) for r in rows]
    power = [float(r[3].split()[0]) for r in rows]
    util = [int(r[4].split()[0]) for r in rows]
    vram = [int(r[5].split()[0]) for r in rows]

    print(f"\ngpu samples       {len(rows)}")
    print(f"temperature       {min(temp)} to {max(temp)} C, {sum(1 for t in temp if t > 88)} above 88")
    print(f"sm clock          {min(clock):,} to {max(clock):,} MHz")
    print(f"power draw        {min(power)} to {max(power)} W")
    print(f"utilization       {min(util)} to {max(util)} %")
    print(f"vram used         {min(vram):,} to {max(vram):,} MiB")


if __name__ == "__main__":
    args = parse_args()

    config_path = os.path.join(args.run_dir, "run_config.json")

    if os.path.exists(config_path):
        config = json.load(open(config_path))
        print(f"commit            {config['git_commit']}, dirty {config['git_dirty']}")
        print(f"started           {config['started_at']}")
        print(f"total_steps       {config['total_steps']:,}")
        print(f"device            {config['device_name']}")
        print()

    report_episodes(os.path.join(args.run_dir, "episodes.csv"))
    report_gpu(os.path.join(args.run_dir, "gpu_log.csv"))
