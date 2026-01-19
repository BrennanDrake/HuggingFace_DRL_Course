import argparse
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--tag", type=str, default="charts/episodic_return", help="scalar tag to plot")
    parser.add_argument("--run", type=str, default=None, help="specific run subdir to plot")
    parser.add_argument("--window", type=int, default=50, help="moving average window (0 disables)")
    parser.add_argument("--out", type=str, default=None, help="output image path (png)")
    return parser.parse_args()


def load_scalars(event_files, tag):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        raise RuntimeError("tensorboard is required for this script") from exc

    points = []
    for event_file in event_files:
        ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for ev in ea.Scalars(tag):
            points.append((ev.step, ev.value))
    points.sort(key=lambda x: x[0])
    return points


def moving_average(values, window):
    if window <= 1:
        return values
    out = []
    cumsum = 0.0
    for i, v in enumerate(values):
        cumsum += v
        if i >= window:
            cumsum -= values[i - window]
            out.append(cumsum / window)
        else:
            out.append(cumsum / (i + 1))
    return out


def main():
    args = parse_args()
    if args.run:
        run_dir = os.path.join(args.logdir, args.run)
        pattern = os.path.join(run_dir, "**", "events.out.tfevents.*")
    else:
        pattern = os.path.join(args.logdir, "**", "events.out.tfevents.*")

    event_files = glob.glob(pattern, recursive=True)
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under {args.logdir}")

    points = load_scalars(event_files, args.tag)
    if not points:
        raise SystemExit(f"No scalar data for tag '{args.tag}'")

    steps, values = zip(*points)
    values = list(values)
    values_smoothed = moving_average(values, args.window)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    plt.figure(figsize=(10, 4))
    plt.plot(steps, values, alpha=0.3, label="raw")
    if args.window > 1:
        plt.plot(steps, values_smoothed, label=f"ma{args.window}")
    plt.title(args.tag)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
