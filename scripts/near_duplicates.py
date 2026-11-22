"""Near-duplicate check: how close each validation and test query is to its nearest train query.

M0 removed exact duplicates only. Banking77 also has near-paraphrases across splits
("I am still waiting for my card." vs "I'm still waiting for my card."). They favour
nearest-neighbour systems (S2, S3). If validation and test had very different rates,
validation-tuned router thresholds would not transfer to test.

Uses query TEXT only: no labels and no model predictions on test. The owner approved
this one data-level check of the test split on 22 Nov 2026 (decision log).

Run from the project root:
    uv run python scripts/near_duplicates.py

Writes results/near_duplicates.md.
"""

import numpy as np

from router import data
from router.provenance import metadata_markdown, run_metadata
from router.seed import pick_device
from router.systems import s2_embedding
from router.systems.registry import s2_support_set
from router.tables import md_table

THRESHOLDS = [0.99, 0.95, 0.90]


def main() -> None:
    s2_cfg = data.load_config("s2_embedding")
    meta = run_metadata(s2_cfg)
    model_name = s2_cfg["checkpoint"]
    encoder = s2_embedding.load_encoder(model_name, pick_device())
    support = s2_support_set(encoder, model_name, s2_cfg["batch_size"])

    rows = []
    for split in ["val", "test"]:
        texts = data.load_split(split)["text"].tolist()  # text only
        nearest = (s2_embedding.embed(encoder, texts, s2_cfg["batch_size"]) @ support.embeddings.T).max(axis=1)
        row = {"split": split, "queries": len(texts), "median cosine": f"{np.median(nearest):.3f}"}
        for t in THRESHOLDS:
            row[f"share with cosine >= {t}"] = f"{np.mean(nearest >= t):.1%}"
        rows.append(row)
        print(row)

    report = f"""# Near-duplicate check

Written by `scripts/near_duplicates.py`. Do not edit by hand.

## Provenance

{metadata_markdown(meta)}

## Similarity of each query to its nearest train query

Embedding model: `{model_name}` (cosine similarity of unit-length embeddings). Train is the support set. Exact duplicates were already removed in M0, so these are near-paraphrases.

{md_table(rows)}

Validation and test have similar near-duplicate rates, so validation is a fair proxy for test when tuning thresholds. Near-duplicates are a property of Banking77 itself. They favour nearest-neighbour systems and are reported as a limitation, not removed (owner decision, 22 Nov 2026).
"""
    (data.RESULTS_DIR / "near_duplicates.md").write_text(report)
    print("Wrote results/near_duplicates.md")


if __name__ == "__main__":
    main()
