from engine.cli import main
from tests.conftest import CONFIG_1B, CONFIG_8B


def test_info_1b_fits_on_8_gib(capsys):
    assert main(["info", "--model", str(CONFIG_1B), "--gpu-mem", "8"]) == 0
    out = capsys.readouterr().out
    assert "1.24 B" in out
    assert "does not fit" not in out


def test_info_8b_bf16_does_not_fit_on_8_gib(capsys):
    assert main(["info", "--model", str(CONFIG_8B), "--gpu-mem", "8"]) == 0
    lines = capsys.readouterr().out.splitlines()
    bf16_line = next(line for line in lines if line.strip().startswith("bf16"))
    assert "does not fit" in bf16_line
