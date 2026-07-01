def format_ticks(num):
    """Formats numbers to 1 decimal place with 'k', 'M', or 'B' suffixes."""
    num_to_ret = 0.0
    if num >= 1_000_000_000:
        num = num / 1_000_000_000
        if num.is_integer():
            num_to_ret = f"{int(num)}B"
        else:
            num_to_ret = f"{num:.1f}B"
    elif num >= 1_000_000:
        num = num / 1_000_000
        if num.is_integer():
            num_to_ret = f"{int(num)}M"
        else:
            num_to_ret = f"{num:.1f}M"
    elif num >= 1_000:
        num = num / 1_000
        if num.is_integer():
            num_to_ret = f"{int(num)}k"
        else:
            num_to_ret = f"{num:.1f}k"
    else:
        num_to_ret = num

    return num_to_ret


def enforce_monotonic_increasing(runs, x_key, y_key):
    """
    Filters a list of run dictionaries to ensure y_vals strictly increase
    or stay flat as x_vals increase. Logs what gets dropped.
    """
    if not runs:
        return []

    # Sort primarily by the x-axis value
    sorted_runs = sorted(runs, key=lambda r: r[x_key])
    filtered_runs = [sorted_runs[0]]
    current_max_y = sorted_runs[0][y_key]

    for run in sorted_runs[1:]:
        if run[y_key] >= current_max_y:
            filtered_runs.append(run)
            current_max_y = run[y_key]
        else:
            print(
                f"[DEBUG - Monotonic Drop] {run['method'].upper()} on {run['dataset']} (k={run['k']}, config={run['config']}) "
                f"dropped. {y_key} was {run[y_key]:.4f} but current max is {current_max_y:.4f} at {x_key}={run[x_key]:.2f}")

    return filtered_runs