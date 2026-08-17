"""
One-off migration for conditions generated before generate_conditions.py's
iter0/ layout bug was fixed.

Earlier versions wrote each condition's degraded input under
``<cond_dir>/iter0/`` instead of directly in ``<cond_dir>/``. That's not what
``miss-alignment train`` expects: ``training_directory`` needs its .xml files
flat at its root, and creates ``iter0/``, ``iter1/``, ... itself as backup
snapshots during training -- with the old layout, training_directory would
have had nothing to train on. This moves already-generated conditions from
the old to the new layout in place, without re-downloading or regenerating
anything.

Safe to re-run: conditions already in the new layout (or without a
generate_conditions.py-created iter0/ at all) are left untouched.

Usage:
    python migrate_condition_layout.py [--settings settings.yaml]
    python migrate_condition_layout.py --output-root /path/to/output_root
    python migrate_condition_layout.py --dry-run   # show what would move
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def migrate_condition(cond_dir: Path, dry_run: bool) -> None:
    old_iter0 = cond_dir / "iter0"
    if not old_iter0.is_dir():
        return

    if list(cond_dir.glob("*.xml")):
        # Flat layout already has input xmls -- either already migrated, or
        # this iter0/ is miss-alignment train's own post-training backup.
        # Leave it alone either way.
        return

    if (old_iter0 / "model.ckpt").exists():
        print(
            f"[skip] {cond_dir.name}: iter0/ contains a model.ckpt -- looks "
            "like training actually ran against the old layout; check by hand."
        )
        return

    xml_files = sorted(old_iter0.glob("*.xml"))
    if not xml_files:
        return

    print(
        f"[migrate] {cond_dir.name}: moving {len(xml_files)} tilt-series out of iter0/"
    )
    for item in sorted(old_iter0.iterdir()):
        destination = cond_dir / item.name
        if dry_run:
            print(f"    {item} -> {destination}")
            continue
        shutil.move(str(item), str(destination))

    if not dry_run and not any(old_iter0.iterdir()):
        old_iter0.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings", type=Path, default=Path(__file__).parent / "settings.yaml"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output_root from settings.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would move, change nothing"
    )
    args = parser.parse_args()

    if args.output_root is not None:
        output_root = args.output_root.expanduser().resolve()
    else:
        settings = yaml.safe_load(args.settings.read_text())
        output_root = Path(settings["output_root"]).expanduser().resolve()

    if not output_root.is_dir():
        sys.exit(f"output_root not found: {output_root}")

    for cond_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        migrate_condition(cond_dir, args.dry_run)


if __name__ == "__main__":
    main()
