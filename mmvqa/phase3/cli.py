"""Run a reproducible Phase 3 metadata check or mounted-image smoke check."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .benchmark import DEFAULT_VARIANTS, S8Benchmark
from .images import ImageResolver
from .storage import S8Store


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".phase3-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(config: dict, *, metadata_only: bool = False, checkout: str | None = None) -> dict:
    """Save a fresh report; missing/corrupt pixels cannot produce a pass."""
    output = Path(config.get("output_dir", "outputs/phase3/local/bootstrap"))
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "mode": "metadata_only" if metadata_only else "image_smoke",
        "training_ready": False,
        "preview_only": True,
        "samples": [],
    }
    write_json(output / "sample_report.json", report)
    write_json(output / "evaluator_records.json", [])
    write_json(output / "provenance.json", [])
    store = S8Store(
        repo=config.get("repository", "bogdanparvu18/msc-graduate-project"),
        ref=config.get("ref", "main"),
        cache_dir=config.get("cache_dir", ".cache/mmvqa-phase3"),
        token=os.environ.get("GITHUB_TOKEN"),
        checkout=checkout,
    )
    try:
        benchmark = S8Benchmark(store)
        examples = benchmark.sample(
            variants=config.get("variants", DEFAULT_VARIANTS),
            per_variant=config.get("per_variant", 1),
        )
        source_records = benchmark.source_records(examples)
        resolver = None if metadata_only else ImageResolver(
            original_roots=config.get("original_roots", []),
            archive_root=config.get("archive_root"),
            derived_roots=config.get("derived_roots", []),
            original_overrides=config.get("original_overrides"),
            archive_manifest=benchmark.archive_manifest,
            source_records=source_records,
        )
        evaluators, provenance = [], []
        for example in examples:
            if resolver is None:
                inspected = benchmark.inspect(example)
            else:
                inspected = benchmark.prepare(
                    example, resolver,
                    question=config.get("preview_question", "Describe the visible image content. This is a loading preview."),
                )
            evaluators.append(inspected["evaluator"])
            provenance.append(inspected["provenance"])
            report["samples"].append({
                "example_id": example["example_id"],
                "subject_id": example["subject_id"],
                "variant": example["variant"],
                "annotation_assisted": example["annotation_assisted"],
                "image_count": example["image_count"],
                "pixels_verified_now": resolver is not None,
                "image_sizes": [list(image.size) for image in inspected["model"]["images"]] if resolver else [],
            })
        write_json(output / "evaluator_records.json", evaluators)
        write_json(output / "provenance.json", provenance)
        report.update(
            status="metadata_verified_images_not_checked" if metadata_only else "image_smoke_passed",
            resolved_commit=store.resolve(),
            processing_signature=benchmark.catalog["processing_signature"],
        )
        return report
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "source_snapshot.json", store.snapshot())
        write_json(output / "sample_report.json", report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase3/bootstrap.json")
    parser.add_argument("--ref", help="GitHub branch/tag or exact commit; default comes from config (main)")
    parser.add_argument("--metadata-only", action="store_true", help="Validate metadata without claiming image availability")
    parser.add_argument("--checkout", help="Reuse immutable Git objects from this local checkout after live GitHub resolution")
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.ref:
            config["ref"] = args.ref
        report = run(config, metadata_only=args.metadata_only, checkout=args.checkout)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Phase 3 check failed: {error}")
        return 1
    print(f"{report['status']}: {len(report['samples'])} samples; commit {report['resolved_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
