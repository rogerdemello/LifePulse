"""
WSGI entry point for production deployment
"""
from app.app import create_app

# Create the Flask app instance
app = create_app()

if __name__ == "__main__":
    app.run()
