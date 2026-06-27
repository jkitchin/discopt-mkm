"""The bundled skill and its installer."""

from discopt_mkm import install


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
