import threading
from flask import Flask
from flask_cors import CORS
from routers.index import registerRoutes
from dotenv import load_dotenv
from websockets.ws import createWebSocket

def create_app():
    # Load environment variables
    load_dotenv()
    
    app = Flask(__name__)

    # Basic config
    app.config['DEBUG'] = False
    
    # Configure CORS
    CORS(app, resources={ 
        r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173", 
                              "http://localhost:3001", "http://127.0.0.1:3001"]},
        r"/socket.io/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173",
                                     "http://localhost:3001", "http://127.0.0.1:3001"]}
    })
    
    # Configure WebSocket client
    socketio = createWebSocket(app)

    # Register routes from external module
    registerRoutes(app, "/api")

    return app, socketio


if __name__ == '__main__':
    app, socketio = create_app()
    
    socketio.run(app, host='0.0.0.0', port=3001)
