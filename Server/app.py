from flask import Flask
from flask_cors import CORS
from routers.index import registerRoutes

def create_app():
    app = Flask(__name__)

    # Basic config
    app.config['DEBUG'] = False
    
    # Configure CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173"
            ]
        }
    })

    # Register routes from external module
    registerRoutes(app, "/api")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=3001)
