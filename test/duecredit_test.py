import pytest
import os
import sys
import subprocess as sp
from datetime import datetime


@pytest.fixture
def script_path():
    yield os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "simple_script.py"
    )


@pytest.fixture
def dc_context(monkeypatch):
    """Backup and delete duecredit cache, replacing it after the test, and enable duecredit"""
    original = os.path.join(os.getcwd(), ".duecredit.p")
    timestamp = datetime.utcnow().isoformat()
    if os.name == "nt":
        timestamp.replace(':', ';')
    backup = original + '.' + timestamp
    if os.path.isfile(original):
        os.rename(original, backup)

    monkeypatch.setenv("DUECREDIT_ENABLE", "yes")

    yield

    if os.path.isfile(original):
        os.remove(original)
    if os.path.isfile(backup):
        os.rename(backup, original)


def test_duecredit(script_path, dc_context):
    result = sp.Popen([sys.executable, script_path])
    result.wait()
    assert result.returncode == 0

    result2 = sp.Popen(["duecredit", "summary", "--format", "bibtex"], stdout=sp.PIPE)
    result2.wait()
    assert result2.returncode == 0

    output = result2.stdout.read().decode("utf-8")
    for author in [
        "LaPlante",
        "Latora",
        "Onnela",
        "Fagiolo",
        "Rubinov",
        "Leicht",
        "Reichardt",
        "Good",
        "Maslov",
    ]:
        assert author in output

    # fails due to bug 152
    # assert output.count('@') == 10

    headers = {line for line in output.split('\n') if line.startswith("@")}
    assert len(headers) == 10
