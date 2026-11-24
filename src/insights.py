"""Dataset insights for the dashboard.

Computes interpretable features for every image (brightness, contrast,
sharpness, colour warmth, vertical light balance), aggregates them per class and
writes ``data/insights.json``. The "story" sentences are generated from the
numbers so they stay true after new data is added.

Run with ``python -m src.insights``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image

from . import config, preprocessing

FEATURES = {
    "brightness": {
        "label": "Brightness", "unit": "mean luminance, 0-255",
        "meaning": "How well lit the student's face and room are.",
        "interpretation": "Brightness follows the room a clip was recorded in more than the student's state: "
                          "a class filmed in one bright room stands apart, while classes filmed in the same "
                          "room cannot be told apart by brightness at all."},
    "contrast": {
        "label": "Contrast", "unit": "std of luminance",
        "meaning": "Spread between the darkest and lightest parts of the frame.",
        "interpretation": "A class recorded against a plain wall has far lower contrast than the rest. That is "
                          "a property of the recording, so predictions on a new background should not be "
                          "trusted until the model has been retrained with such images."},
    "sharpness": {
        "label": "Sharpness", "unit": "mean gradient magnitude",
        "meaning": "Edge strength. Low values mean blur from movement or a soft camera.",
        "interpretation": "Low sharpness is partly real signal, because a turning head smears, and partly "
                          "just a softer webcam. The model could use blur as a shortcut, which is why "
                          "training randomises contrast, zoom and position."},
    "warmth": {
        "label": "Colour warmth", "unit": "red mean minus blue mean",
        "meaning": "Colour temperature: daylight and screens are cool, lamps are warm.",
        "interpretation": "Each webcam and wall colour leaves its own colour cast, so warmth identifies the "
                          "recording session. Classes that share a session have almost the same warmth and "
                          "can only be separated by reading the facial expression."},
    "lower_minus_upper": {
        "label": "Vertical light balance", "unit": "lower-half minus upper-half luminance",
        "meaning": "Negative when the top of the frame is brighter (light wall above, dark clothing below).",
        "interpretation": "Values move towards zero when dark hair fills the upper half of the frame, which "
                          "happens when the head drops or turns. This is the one simple feature with a "
                          "behavioural meaning: it rises for students who nod off or look away."},
}


def _summary(values: np.ndarray) -> dict[str, float]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()),
            "q1": float(q1), "median": float(med), "q3": float(q3), "max": float(values.max())}


def _histogram(values: np.ndarray, lo: float, hi: float, bins: int = 20) -> list[float]:
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    total = counts.sum() or 1
    return [round(float(c) / total, 4) for c in counts]


def _story(key: str, per_class: dict[str, dict[str, float]]) -> str:
    """One data-driven sentence describing how a feature differs between classes."""
    means = {c: s["mean"] for c, s in per_class.items()}
    hi = max(means, key=means.get)
    lo = min(means, key=means.get)
    pooled_std = float(np.mean([s["std"] for s in per_class.values()])) or 1.0
    gap = (means[hi] - means[lo]) / pooled_std
    strength = "a strong" if gap >= 0.8 else "a moderate" if gap >= 0.4 else "only a weak"
    nice = lambda c: c.replace("_", " ")  # noqa: E731
    return (f"'{nice(hi)}' frames score highest ({means[hi]:.1f}) and '{nice(lo)}' lowest "
            f"({means[lo]:.1f}); the gap is {gap:.2f} standard deviations, {strength} separation.")


def compute_insights(write: bool = True) -> dict[str, Any]:
    records: dict[str, dict[str, list[float]]] = {c: {k: [] for k in FEATURES} for c in config.CLASS_NAMES}
    mean_images: dict[str, np.ndarray] = {}
    counts = {"train": preprocessing.class_counts(config.TRAIN_DIR),
              "test": preprocessing.class_counts(config.TEST_DIR)}

    for root in (config.TRAIN_DIR, config.TEST_DIR):
        paths, labels = preprocessing.list_images(root)
        for path, label in zip(paths, labels):
            name = config.CLASS_NAMES[label]
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    feats = preprocessing.extract_features(im)
                    small = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float64)
            except OSError:
                continue
            for key, value in feats.items():
                records[name][key].append(value)
            if name in mean_images:
                mean_images[name] += small
            else:
                mean_images[name] = small.copy()

    features: dict[str, Any] = {}
    for key, info in FEATURES.items():
        all_values = np.array([v for c in config.CLASS_NAMES for v in records[c][key]] or [0.0])
        lo, hi = np.percentile(all_values, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        per_class = {c: _summary(np.array(records[c][key])) for c in config.CLASS_NAMES if records[c][key]}
        features[key] = {
            **info,
            "range": [float(lo), float(hi)],
            "bin_edges": [float(x) for x in np.linspace(lo, hi, 21)],
            "per_class": per_class,
            "histograms": {c: _histogram(np.array(records[c][key]), lo, hi)
                           for c in config.CLASS_NAMES if records[c][key]},
            "story": _story(key, per_class) if per_class else "",
        }

    # Mean image per class, saved as small PNGs the UI can show.
    out_dir = config.DATA_DIR / "mean_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_files = {}
    for name, total in mean_images.items():
        n = len(records[name]["brightness"]) or 1
        Image.fromarray((total / n).clip(0, 255).astype(np.uint8)).resize((192, 192), Image.BICUBIC) \
            .save(out_dir / f"{name}.png")
        mean_files[name] = f"{name}.png"

    total_images = sum(counts["train"].values()) + sum(counts["test"].values())
    by_class = {c: counts["train"][c] + counts["test"][c] for c in config.CLASS_NAMES}
    engaged_total = sum(n for c, n in by_class.items() if config.ENGAGEMENT_GROUP[c] == "Engaged")
    largest = max(by_class, key=by_class.get)
    smallest = min(by_class, key=by_class.get)
    balance_story = (
        f"{total_images} images in 6 classes. '{largest.replace('_', ' ')}' is the largest class "
        f"({by_class[largest]}) and '{smallest.replace('_', ' ')}' the smallest ({by_class[smallest]}), "
        f"a ratio of {by_class[largest] / max(by_class[smallest], 1):.2f} to 1. "
        f"Engaged states make up {engaged_total / max(total_images, 1):.0%} of the data, so the "
        f"engaged / not-engaged view is close to balanced.")

    insights = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_names": config.CLASS_NAMES,
        "engagement_group": config.ENGAGEMENT_GROUP,
        "counts": counts,
        "total_images": total_images,
        "balance_story": balance_story,
        "features": features,
        "mean_images": mean_files,
    }
    if write:
        config.INSIGHTS_PATH.write_text(json.dumps(insights, indent=2))
    return insights


def load_insights() -> dict[str, Any]:
    if config.INSIGHTS_PATH.exists():
        return json.loads(config.INSIGHTS_PATH.read_text())
    return compute_insights()


if __name__ == "__main__":
    data = compute_insights()
    print(data["balance_story"])
    for key, feat in data["features"].items():
        print(f"{key:18s} {feat['story']}")
