import os
from flask import Flask, render_template
from dotenv import load_dotenv

load_dotenv()

from app.routes.sleep import sleep_bp
from app.routes.nutrition import nutrition_bp
from app.routes.calculator_routes import calculator_bp
from app.routes.heart import heart_disease_bp
from app.routes.migraine import migraine_bp
from app.routes.health_score import health_score_bp

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')

    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-default-secret-key")
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.cache = {}

    # Register Blueprints
    app.register_blueprint(sleep_bp)
    app.register_blueprint(nutrition_bp)
    app.register_blueprint(calculator_bp)
    app.register_blueprint(heart_disease_bp)
    app.register_blueprint(migraine_bp)
    app.register_blueprint(health_score_bp)

    @app.route('/')
    def index():
        return render_template('index.html')

    return app