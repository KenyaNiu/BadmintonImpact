"""Command-line surface checks."""

import pytest

from badminton_impact_ai.cli import main


@pytest.mark.parametrize("command", ["prepare", "run", "analyze", "artifacts"])
def test_subcommand_help(command: str) -> None:
    with pytest.raises(SystemExit) as error:
        main([command, "--help"])
    assert error.value.code == 0
