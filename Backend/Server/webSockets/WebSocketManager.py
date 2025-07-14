from flask_socketio import SocketIO, emit
from flask import request

class WebSocketManager:
    def __init__(self, app):
        self.socketio = SocketIO(cors_allowed_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173", 
            "http://localhost:3001", "http://127.0.0.1:3001"
        ])
        self.socketio.init_app(app)
        self.connected_clients = []

        self.register_handlers()

    def register_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            self.connected_clients.append(request.sid)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            if request.sid in self.connected_clients:
                self.connected_clients.remove(request.sid)

    def broadcast(self, data):
        """Send a message to all connected clients."""
        for sid in self.connected_clients:
            self.socketio.emit('message', data, room=sid)

    def run(self, app, host='0.0.0.0', port=3001):
        self.socketio.run(app, host=host, port=port)
