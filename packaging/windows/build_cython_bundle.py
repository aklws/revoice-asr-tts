from __future__ import annotations

import argparse
import ast
import os
import shutil
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


PACKAGE_SPECS: tuple[tuple[str, str], ...] = (
    ("app", "app"),
    ("ui", "ui"),
)
CYTHON_EXCLUDE_RELATIVE_PATHS: set[str] = set()
IGNORED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
}
IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def _copy_package_tree(source_dir: Path, target_dir: Path) -> None:
    def _ignore(_directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in IGNORED_DIR_NAMES:
                ignored.add(name)
                continue
            if Path(name).suffix in IGNORED_SUFFIXES:
                ignored.add(name)
        return ignored

    shutil.copytree(source_dir, target_dir, ignore=_ignore)


def _requires_source_fallback(py_file: Path) -> bool:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    return any(isinstance(node, ast.Match) for node in ast.walk(tree))


def _detect_source_fallbacks(bundle_root: Path) -> set[str]:
    fallback_paths: set[str] = set(CYTHON_EXCLUDE_RELATIVE_PATHS)
    for py_file in sorted(bundle_root.rglob("*.py")):
        relative_posix = py_file.relative_to(bundle_root).as_posix()
        try:
            requires_fallback = _requires_source_fallback(py_file)
        except SyntaxError:
            requires_fallback = True
        if requires_fallback:
            fallback_paths.add(relative_posix)
    return fallback_paths


def _collect_extensions(bundle_root: Path, excluded_relative_paths: set[str]) -> list[Extension]:
    extensions: list[Extension] = []
    for py_file in sorted(bundle_root.rglob("*.py")):
        relative_posix = py_file.relative_to(bundle_root).as_posix()
        if relative_posix in excluded_relative_paths:
            continue
        module_name = ".".join(py_file.relative_to(bundle_root).with_suffix("").parts)
        extensions.append(Extension(module_name, [relative_posix]))
    return extensions


def _strip_python_sources(bundle_root: Path, excluded_relative_paths: set[str]) -> None:
    for path in sorted(bundle_root.rglob("*"), reverse=True):
        if path.is_dir():
            if path.name in IGNORED_DIR_NAMES:
                shutil.rmtree(path, ignore_errors=True)
            continue
        if path.suffix in {".py", ".c", ".html"}:
            relative_posix = path.relative_to(bundle_root).as_posix()
            if path.suffix == ".py" and relative_posix in excluded_relative_paths:
                continue
            path.unlink(missing_ok=True)


def build_bundle(repo_root: Path, output_root: Path, build_root: Path, dry_run: bool) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    copied_targets: list[Path] = []
    for source_rel, target_rel in PACKAGE_SPECS:
        source_dir = repo_root / source_rel
        target_dir = output_root / target_rel
        if not source_dir.exists():
            raise FileNotFoundError(f"source directory does not exist: {source_dir}")
        _copy_package_tree(source_dir, target_dir)
        copied_targets.append(target_dir)

    source_fallbacks = _detect_source_fallbacks(output_root)
    extensions = _collect_extensions(output_root, source_fallbacks)
    if dry_run:
        print(f"bundle_root={output_root}")
        print(f"copied_packages={len(copied_targets)}")
        print(f"extension_count={len(extensions)}")
        print(f"excluded_extensions={len(source_fallbacks)}")
        return

    current_dir = Path.cwd()
    os.chdir(output_root)
    try:
        ext_modules = cythonize(
            extensions,
            compiler_directives={
                "language_level": "3",
                "binding": True,
            },
            nthreads=max((os.cpu_count() or 1) - 1, 1),
            build_dir="cy",
        )

        setup(
            name="revoice-cython-bundle",
            script_args=[
                "build_ext",
                "--inplace",
                "--build-temp=tmp",
            ],
            ext_modules=ext_modules,
            zip_safe=False,
        )
    finally:
        os.chdir(current_dir)

    for artifact_dir in (output_root / "cy", output_root / "tmp"):
        shutil.rmtree(artifact_dir, ignore_errors=True)

    _strip_python_sources(output_root, source_fallbacks)

    compiled_extensions = list(output_root.rglob("*.pyd"))
    print(f"bundle_root={output_root}")
    print(f"compiled_extensions={len(compiled_extensions)}")
    print(f"source_fallbacks={len(source_fallbacks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local source packages into Cython pyd modules.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_bundle(
        repo_root=args.repo_root.resolve(),
        output_root=args.output_root.resolve(),
        build_root=args.build_root.resolve(),
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
