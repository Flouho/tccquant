from pathlib import Path

from tccquant.ppq_cleanup import prune_ppq_tree


def test_prune_ppq_tree_dry_run(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "ppq").mkdir()
    removed = prune_ppq_tree(tmp_path, dry_run=True)
    assert any(p.endswith("tests") for p in removed)
    assert any(p.endswith("docs") for p in removed)
    assert not any(p.endswith("ppq") for p in removed)
