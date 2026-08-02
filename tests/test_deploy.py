"""The deploy runs on the Python the pinned wheels were built for.

This file exists because a build failed for a week's worth of pushes while the
repository looked correctly configured. `runtime.txt` said `python-3.12.7`, CI
said 3.12 and cited runtime.txt, and Render built on 3.13.4 -- because Render
does not read runtime.txt. It reads a `PYTHON_VERSION` environment variable or
a `.python-version` file, and with neither present it falls back to a default
that depends on when the service was created.

Nothing was wrong with the pin. The pin was being read by nobody.

That matters more than a version mismatch usually does: numpy 1.26 and pandas
2.1 publish no cp313 wheels, so pip fell back to compiling pandas from source
against 3.13 headers, where Cython-generated calls to `_PyLong_AsByteArray`
no longer match the signature. The build did not degrade, it failed. And those
versions are not free to bump -- requirements.txt pins them to what the model
artifacts in app/models/ were trained with.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / ".python-version"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _pinned():
    assert PIN.exists(), (
        ".python-version is missing. Render reads this file (or a "
        "PYTHON_VERSION env var) and otherwise picks a default -- it does not "
        "read runtime.txt."
    )
    text = PIN.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", text), (
        f".python-version must hold a bare version and nothing else, got {text!r}"
    )
    return text


def _minor(version):
    return ".".join(version.split(".")[:2])


def test_the_deploy_python_version_is_pinned_where_the_platform_reads_it():
    _pinned()


def test_ci_tests_the_version_the_deploy_will_use():
    """CI passing on 3.12 said nothing about a deploy building on 3.13."""
    ci = CI.read_text(encoding="utf-8")
    found = re.search(r"python-version:\s*['\"]?([\d.]+)['\"]?", ci)
    assert found, "the CI workflow no longer pins a Python version"
    assert _minor(found.group(1)) == _minor(_pinned()), (
        f"CI runs {found.group(1)} but the deploy will use {_pinned()}"
    )


def test_the_interpreter_running_these_tests_matches_the_pin():
    """The pinned wheels are per-minor-version; a green suite on a different
    one is not evidence the deploy will install.
    """
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert running == _minor(_pinned()), (
        f"these tests are running on {running} but the deploy pins "
        f"{_pinned()}; requirements.txt has no wheels for {running}"
    )


def test_no_second_file_claims_to_set_the_python_version():
    """runtime.txt is the Heroku convention and Render ignores it. Leaving one
    in the tree is how this went unnoticed: it read as a pin, CI's comment
    pointed at it, and it governed nothing.
    """
    stale = ROOT / "runtime.txt"
    if not stale.exists():
        return
    declared = stale.read_text(encoding="utf-8").strip().removeprefix("python-")
    assert _minor(declared) == _minor(_pinned()), (
        f"runtime.txt says {declared} and .python-version says {_pinned()}. "
        "Only the latter is read by the deploy; delete runtime.txt or keep "
        "them in step."
    )
