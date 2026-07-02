"""Install the bundled Claude skill (and optionally register the MCP server).

Console script ``discopt-mkm-install-skill``: copies the packaged ``SKILL.md`` into
a Claude skills directory so the skill is auto-discovered, and optionally runs
``claude mcp add`` to register the MCP server.
"""

from __future__ import annotations

import argparse
import subprocess
from importlib import resources
from pathlib import Path

SKILL_NAME = "discopt-mkm"
MCP_ADD = ["claude", "mcp", "add", SKILL_NAME, "--", "discopt-mkm-mcp"]


def _bundled_skill() -> str:
    """Read the packaged SKILL.md text."""
    return resources.files("discopt.mkm").joinpath("skill", "SKILL.md").read_text()


def install_skill(skills_dir: Path, force: bool = False) -> Path:
    """Copy the bundled SKILL.md into ``skills_dir/discopt-mkm/SKILL.md``."""
    dest = Path(skills_dir) / SKILL_NAME / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"skill already installed: {dest}\n  (use --force to overwrite)")
        return dest
    dest.write_text(_bundled_skill())
    print(f"installed skill -> {dest}")
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="discopt-mkm-install-skill",
        description="Install the discopt-mkm Claude skill (and optionally the MCP server).",
    )
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--user", action="store_true", help="install to ~/.claude/skills (default)")
    scope.add_argument("--project", action="store_true", help="install to ./.claude/skills (versioned with a repo)")
    p.add_argument("--force", action="store_true", help="overwrite an existing skill")
    p.add_argument("--mcp", action="store_true", help="also register the MCP server via `claude mcp add`")
    args = p.parse_args(argv)

    skills_dir = Path(".claude/skills") if args.project else Path.home() / ".claude" / "skills"
    install_skill(skills_dir, force=args.force)

    if args.mcp:
        print("registering MCP server:", " ".join(MCP_ADD))
        try:
            subprocess.run(MCP_ADD, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"could not run `claude mcp add` ({type(e).__name__}); register it yourself:\n  " + " ".join(MCP_ADD))
    else:
        print("to register the MCP server:  " + " ".join(MCP_ADD))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
