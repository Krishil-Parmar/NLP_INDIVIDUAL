"""Small helper for writing markdown tables into result files."""


def md_table(rows: list[dict]) -> str:
    """Turn a list of dicts (all with the same keys) into a markdown table."""
    if not rows:
        return "(no rows)"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)
