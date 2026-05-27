def format_ticks(num):
    """Formats numbers to 1 decimal place with 'k' or 'M' suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}k"
    else:
        return f"{num:.1f}"