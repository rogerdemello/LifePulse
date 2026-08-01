import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"


@pytest.fixture(scope="session")
def app():
    from app.app import create_app

    application = create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    """A test client with throttling off.

    Several tests submit dozens of assessments in a second, which is precisely
    what the limiter exists to stop. Tests covering the limiter itself turn it
    back on explicitly.
    """
    app.config["RATELIMIT_DISABLED"] = True
    return app.test_client()


@pytest.fixture()
def throttled_client(app):
    from app.ratelimit import reset

    app.config["RATELIMIT_DISABLED"] = False
    reset()
    yield app.test_client()
    reset()
    app.config["RATELIMIT_DISABLED"] = True


def requires_dataset(filename):
    """Skip a test when the (gitignored) training CSV is not present."""
    return pytest.mark.skipif(
        not (DATA / filename).exists(),
        reason=f"{filename} not available; it is gitignored training data",
    )
