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
from app.routes.sleep import sleep_bp
from app.routes.start import start_bp
from app.security import init_app as init_security

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
    init_security(app)

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
        from app.azure_openai import describe_configuration
        from app.ml.bundle import MODEL_NAMES, try_get_model

        models = {name: try_get_model(name) is not None for name in MODEL_NAMES}
        status = 200 if all(models.values()) else 503
        return {
            "ok": status == 200,
            "models": models,
            "azure_openai": describe_configuration(),
        }, status

    # error_page() has existed since the form-validation work, and rate
    # limiting and model-unavailable both render through it -- but it was
    # never wired to Flask's own handlers. So a mistyped URL got Werkzeug's
    # bare "404 Not Found": no navigation, no way back to an assessment, and
    # no medical disclaimer, on a site whose every other page carries one.
    #
    # A 500 mattered more. The assessment routes catch their own exceptions
    # (see prediction_errors), so a 500 here is something genuinely unexpected
    # -- exactly when a person needs a reference to quote rather than a white
    # page with two words on it.

    @app.errorhandler(404)
    def not_found(_):
        from app.routes.support import error_page

        return error_page(
            "That page doesn't exist. It may have moved, or the link may have "
            "been mistyped. Everything LifePulse can look at is listed on the "
            "start page.",
            status=404,
            title="Page not found",
        )

    @app.errorhandler(405)
    def method_not_allowed(_):
        from app.routes.support import error_page

        return error_page(
            "That page can't be reached that way. If you were part-way through "
            "an assessment, start it again from the beginning.",
            status=405,
            title="That didn't work",
        )

    @app.errorhandler(500)
    def server_error(_):
        from app.routes.support import error_page

        return error_page(
            "Something went wrong at our end, and no result was produced. "
            "Nothing you entered was stored. Please try again -- and if it "
            "keeps happening, quote the reference below.",
            status=500,
            title="Something went wrong",
        )

    return app
