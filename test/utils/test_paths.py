import os
import shutil

from src.utils.paths import (
    get_root_dir,
    get_fig_dirs,
    get_results_dir,
)


def test_root_dir_exists():
    """
    The root directory returned by get_root_dir() must exist on disk and must be a
    directory rather than a file.

    This root path is the anchor used throughout the project for:
        • resolving figure output paths (REVTeX / MDPI pipelines),
        • resolving results/ directories for experiment archives,
        • ensuring a consistent and portable filesystem structure.

    A valid root directory is essential for all I/O components in the pipeline.
    """
    root = get_root_dir()
    assert os.path.exists(root)
    assert os.path.isdir(root)


def test_fig_dirs_are_correct():
    """
    get_fig_dirs() must return the two canonical figure directories used by the
    publication pipeline:

        paper/figures/generated

    These directories must:
        • be normalized to POSIX-style paths (no backslashes),
        • be correctly nested inside the project under 'paper/figures/generated'.

    The test verifies that the returned paths conform to this specification.
    """
    dirs = get_fig_dirs()
    assert len(dirs) == 1

    for d in dirs:
        # Paths must use normalized forward slashes
        assert "\\" not in d

        # Both directories must belong to the paper-generation tree
        assert "paper" in d
        assert d.endswith("figures/generated")


def test_results_dir_is_correct(tmp_path, monkeypatch):
    """
    get_results_dir() must resolve to <root>/results, where <root> is whatever
    get_root_dir() returns.

    This test patches get_root_dir() to an isolated temporary directory, ensuring
    that no real project files are touched and that path computation is correct.

    The results directory is used for:
        • storing NPZ experiment archives,
        • reproducibility metadata,
        • scalar and spectral diagnostic outputs.
    """
    fake_root = tmp_path / "project"
    fake_root.mkdir()

    # Normalize temporary root to POSIX format
    monkeypatch.setattr(
        "src.utils.paths.get_root_dir",
        lambda: str(fake_root).replace("\\", "/")
    )

    expected = fake_root / "results"
    result_dir = get_results_dir()

    assert str(expected).replace("\\", "/") == result_dir
