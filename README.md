# Multi-Agent System Project

This project is composed of a Python backend and a web-based frontend.

## Installation and Setup

Follow these steps to get the project running on your local machine.

### 1. Clone the repository

```bash
git clone https://github.com/tomas-oliveira03/ASM.git
cd ASM
```

### 2. Backend Setup

The backend is a Python application that requires a MongoDB instance.

1.  Navigate to the backend directory:
    ```bash
    cd Backend
    ```
2.  Install required Python packages.
3.  Copy `.env.example` to `.env` and fill in your API keys and credentials.
4.  Start a MongoDB server.
5.  Run the system agents setup file:
    ```bash
    python main.py
    ```
6.  Run the backend server:
    ```bash
    python Server/app.py
    ```
    The backend API will be running on `http://localhost:3001`.

### 3. Frontend Setup

The frontend is a web application built with Node.js.

1.  Navigate to the frontend directory from the project root:
    ```bash
    cd Frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Ensure the backend API is running.
4.  Start the development server:
    ```bash
    npm run dev
    ```
5. Open your browser and navigate to `http://localhost:5173/`
