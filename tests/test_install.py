"""The bundled skill and its installer."""

from pathlib import Path

import pytest

from discopt.mkm import install


def test_bundled_skill_has_frontmatter():
    txt = install._bundled_skill()
    assert txt.startswith("---") and "name: discopt-mkm" in txt and "description:" in txt


def test_install_skill_idempotent_and_force(tmp_path):
    dest = install.install_skill(tmp_path)
    assert dest == tmp_path / "discopt-mkm" / "SKILL.md"
    assert dest.read_text().startswith("---")

    # second call without force does not overwrite (and does not raise)
    dest.write_text("EDITED")
    install.install_skill(tmp_path)
    assert dest.read_text() == "EDITED"

    # force restores the bundled content
    install.install_skill(tmp_path, force=True)
    assert dest.read_text().startswith("---")


def test_main_project_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert install.main(["--project"]) == 0
    assert (tmp_path / ".claude" / "skills" / "discopt-mkm" / "SKILL.md").exists()


# knowhow / npx skills discover skills/<name>/SKILL.md at the repo root; it is a
# symlink to the packaged file, so the two can never drift.
_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (_REPO / "skills").is_dir(), reason="not a source checkout")
def test_repo_layout_links_to_packaged_skill():
    skill = _REPO / "skills" / install.SKILL_NAME / "SKILL.md"
    packaged = _REPO / "src" / "discopt" / "mkm" / "skill" / f"{install.SKILL_NAME}.md"
    assert skill.is_symlink() and skill.resolve() == packaged.resolve()
    # exactly one SKILL.md, so knowhow sees no duplicate skill
    found = [*(_REPO / "skills").rglob("SKILL.md"), *(_REPO / "src").rglob("SKILL.md")]
    assert found == [skill]
