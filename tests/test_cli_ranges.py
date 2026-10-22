"""Validate run options before run preparation or browser startup."""

import pytest
from typer.testing import CliRunner

from loki import cli


class RunPreparationReached(Exception):
    pass


@pytest.fixture
def prepared_options(monkeypatch):
    options = []

    def capture_options(value):
        options.append(value)
        raise RunPreparationReached

    monkeypatch.setattr(cli, "prepare_run", capture_options)
    return options


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--duration", "0"),
        ("--duration", "-1"),
        ("-d", "0"),
        ("--fault-rate", "-0.1"),
        ("--fault-rate", "1.1"),
        ("--fault-rate", "-inf"),
        ("--fault-rate", "inf"),
        ("--fault-rate", "nan"),
    ],
)
def test_invalid_run_ranges_stop_before_preparation(flag, value, prepared_options):
    result = CliRunner().invoke(cli.app, ["run", "http://localhost:8000", flag, value])
    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert prepared_options == []


@pytest.mark.parametrize(
    "args,duration,fault_rate",
    [
        ([], None, None),
        (["--duration", "1"], 1, None),
        (["-d", "5"], 5, None),
        (["--fault-rate", "0"], None, 0.0),
        (["--fault-rate", "0.3"], None, 0.3),
        (["--fault-rate", "1"], None, 1.0),
        (["--duration", "1", "--fault-rate", "1"], 1, 1.0),
    ],
)
def test_valid_run_ranges_reach_preparation(args, duration, fault_rate, prepared_options):
    result = CliRunner().invoke(cli.app, ["run", "http://localhost:8000", *args])
    assert isinstance(result.exception, RunPreparationReached)
    assert len(prepared_options) == 1
    assert prepared_options[0].duration == duration
    assert prepared_options[0].fault_rate == fault_rate
