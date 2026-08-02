import logging
import os

from dotenv import load_dotenv
from flask import Flask, render_template

from app.ml.bundle import all_metadata
from app.ml.triage import CONCERNS
from app.observability import init_app as init_observability
from app.routes.calculator_routes import calculator_bp
from app.routes.health_score import health_score_bp
from app.routes.heart import heart_disease_bp
from app.routes.migraine import migraine_bp
from app.routes.nutrition import nutrition_bp
from app.routes.start import start_bp
from app.routes.sleep import sleep_bp

load_dotenv()

log = logging.getLogger(__name__)


def _secret_key():
    """Return the signing key, refusing to run on a placeholder in production.

    This used to fall back to the literal string "your-default-secret-key",
    which would silently ship to production and make every session cookie
    forgeable by anyone who has read this repository.
    """
    key = os.getenv("SECRET_KEY")
    if key:
        return key
    if os.getenv("FLASK_ENV") == "production" or os.getenv("RENDER"):
        raise RuntimeError(
            "SECRET_KEY is not set. Refusing to start in production with a "
            "predictable signing key. Set SECRET_KEY in the environment."
        )
    log.warning(
        "SECRET_KEY is not set; using an ephemeral development key. "
        "Sessions will not survive a restart."
    )
    return os.urandom(32)


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    app.config["SECRET_KEY"] = _secret_key()
    app.config["SENTRY_DSN"] = os.getenv("SENTRY_DSN")
    app.config["ENVIRONMENT"] = os.getenv("FLASK_ENV", "development")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.cache = {}

    app.register_blueprint(sleep_bp)
    app.register_blueprint(nutrition_bp)
    app.register_blueprint(calculator_bp)
    app.register_blueprint(heart_disease_bp)
    app.register_blueprint(migraine_bp)
    app.register_blueprint(health_score_bp)
    app.register_blueprint(start_bp)

    init_observability(app)

    @app.context_processor
    def inject_model_metadata():
        """Expose real model metrics to templates.

        Pages quote their model's accuracy; reading it from metadata.json keeps
        those figures honest instead of frozen at whatever was true when the
        HTML was written.
        """
        from app.azure_openai import is_configured as azure_configured

        return {
            "model_meta": all_metadata(),
            "llm_enabled": azure_configured(),
        }

    @app.route("/")
    def index():
        """The landing page lists the same six concerns /start routes to.

        It used to hard-code its own six cards, with their own titles and
        blurbs, next to a triage page generating its list from CONCERNS. Two
        descriptions of one set of assessments is one more than can be kept
        true, so the page reads the same tuple the router does.
        """
        return render_template("index.html", concerns=CONCERNS)

    @app.route("/privacy")
    def privacy():
        return render_template("privacy.html")

    @app.route("/summary")
    def summary():
        """Printable visit summary.

        Deliberately renders an empty shell: the saved results live in the
        browser's localStorage and are assembled client-side by
        static/js/summary.js. The server never sees them.
        """
        return render_template("summary.html")

    @app.route("/healthz")
    def healthz():
        """Liveness probe that also reports which models loaded."""
        from app.ml.bundle import MODEL_NAMES, try_get_model

        from app.azure_openai import describe_configuration

        models = {name: try_get_model(name) is not None for name in MODEL_NAMES}
        status = 200 if all(models.values()) else 503
        return {
            "ok": status == 200,
            "models": models,
            "azure_openai": describe_configuration(),
        }, status

    return app
