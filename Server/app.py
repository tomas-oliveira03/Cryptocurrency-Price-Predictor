from flask import Flask
from routers.index import registerRoutes

def create_app():
    app = Flask(__name__)

    # Basic config
    app.config['DEBUG'] = False

    # Register routes from external module
    registerRoutes(app, "/api")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=3001)
