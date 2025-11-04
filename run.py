import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(__file__))

from app.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
