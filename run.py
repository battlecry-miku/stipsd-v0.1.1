"""Run masking, completion and optional source regression on a sample dataset."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from stipsd import __version__
from stipsd.data import load_data, split_indices, fit_scaler, transform_images
from stipsd.completion import METHODS, make_mask, complete
from stipsd.metrics import image_metrics, source_metrics

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "examples/sample_data.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/default")
    parser.add_argument("--with-adr", action="store_true", help="Include the source-free ADR proxy correction (requires torch)")
    parser.add_argument("--train-epochs", type=int, default=0, help="Fresh CPU training epochs for source regression; 0 disables source prediction")
    args = parser.parse_args()
    if args.train_epochs < 0:
        parser.error("--train-epochs cannot be negative")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; choose a new --out directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    x_raw, y, ids = load_data(args.data)
    split = split_indices(len(x_raw), config["seed"])
    scaler = fit_scaler(x_raw[split["train"]], y[split["train"]])
    x = transform_images(x_raw, scaler)
    if args.with_adr or args.train_epochs:
        import torch
        torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "split.json", {name: ids[indices].tolist() for name, indices in split.items()})
    write_json(args.out / "scaler.json", scaler)
    write_json(args.out / "config_used.json", config)
    model, best_epoch = None, None
    if args.train_epochs:
        from stipsd.training import train, predict
        model, best_epoch = train(x, y, split, scaler, args.train_epochs, config["seed"], args.out)
        clean = []
        for index in split["test"]:
            clean.append({"sample_id": str(ids[index]), **source_metrics(y[index], predict(model, x[index], scaler), scaler["label_range"])})
        write_csv(args.out / "clean_source_predictions.csv", clean)
    methods = list(METHODS) + (["ns_adr_proxy"] if args.with_adr else [])
    rows, sources, masks, saved_completions, mask_records = [], [], [], [], []
    for scenario_index, scenario in enumerate(config["scenarios"]):
        for rate_index, rate in enumerate(config["missing_rates"]):
            for index in split["test"]:
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], scenario_index, rate_index, int(index)]))
                mask = make_mask((40, 40), scenario, rate, rng)
                mask_id = len(masks)
                masks.append(mask)
                observed = x[index].copy()
                observed[mask] = 0
                common = {"sample_id": str(ids[index]), "scenario": scenario,
                          "requested_rate": rate, "actual_rate": float(mask.mean()), "mask_id": mask_id}
                mask_records.append(common)
                all_outputs = []
                for method in methods:
                    if method == "ns_adr_proxy":
                        from stipsd.physics import correct
                        completed = correct(observed, mask, config["adr_proxy"])
                    else:
                        completed = complete(observed, mask, method, config["svd_rank"], config["svd_iterations"])
                    all_outputs.append(completed)
                    rows.append({**common, "method": method, **image_metrics(x[index], completed, mask)})
                    if model is not None:
                        sources.append({**common, "method": method,
                                        **source_metrics(y[index], predict(model, completed, scaler), scaler["label_range"])})
                saved_completions.append(all_outputs)
    write_csv(args.out / "completion_metrics.csv", rows)
    write_csv(args.out / "mask_manifest.csv", mask_records)
    if sources:
        write_csv(args.out / "source_predictions.csv", sources)
    np.savez_compressed(args.out / "masks.npz", missing=np.stack(masks))
    np.savez_compressed(args.out / "completed_fields.npz", images=np.asarray(saved_completions), methods=np.asarray(methods))
    summary = []
    for scenario in config["scenarios"]:
        for rate in config["missing_rates"]:
            for method in methods:
                group = [r for r in rows if r["scenario"] == scenario and r["requested_rate"] == rate and r["method"] == method]
                summary.append({"scenario": scenario, "requested_rate": rate, "method": method, "n": len(group),
                                **{key: float(np.mean([r[key] for r in group])) for key in
                                   ("actual_rate", "completion_RMSE_all", "completion_RMSE_missing", "completion_SSIM_all")}})
    write_csv(args.out / "summary.csv", summary)
    versions = {}
    for name in ("numpy", "opencv-python-headless", "opencv-python", "scikit-image", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    run = {"package_version": __version__, "python": platform.python_version(), "dependencies": versions,
           "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
           "samples": len(x), "split_counts": {name: len(indices) for name, indices in split.items()},
           "methods": methods, "completion_rows": len(rows), "train_epochs": args.train_epochs,
           "best_epoch": best_epoch, "source_strategy": "B_frozen_clean_model" if model is not None else "not_run",
           "clipped_pixels_fraction": {name: float(np.mean(x_raw[indices] > scaler["image_divisor"])) for name, indices in split.items()}}
    write_json(args.out / "run_manifest.json", run)
    print(json.dumps(run, indent=2))
    print(f"Run complete: {args.out.resolve()}")


if __name__ == "__main__":
    main()
