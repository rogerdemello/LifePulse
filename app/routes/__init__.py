from flask import Blueprint

# Import all individual route modules
from .calculator_routes import calculator_bp
from .nutrition import nutrition_bp
from .sleep import sleep_bp
from .heart import heart_disease_bp

# Create a Blueprint to register all child blueprints if needed
routes = Blueprint('routes', __name__)

# Register all routes
def register_routes(app):
    app.register_blueprint(sleep_bp)
    app.register_blueprint(calculator_bp)
    app.register_blueprint(nutrition_bp)
    app.register_blueprint(heart_disease_bp)

