# Legalify: Your AI-Powered Legal Document Assistant

## Problem Statement
Legal documents are notoriously complex, dense, and time-consuming to understand. Lawyers, legal professionals, and even individuals often spend countless hours sifting through intricate legal texts to find specific information, understand clauses, or extract relevant details. This process is inefficient, prone to human error, and creates a significant barrier to accessing and comprehending legal information.

## Our Solution
Legalify addresses this challenge by leveraging the power of Retrieval-Augmented Generation (RAG) models on Google Cloud Vertex AI. It provides an intuitive, interactive platform where users can upload legal documents and ask natural language questions. Our RAG pipeline intelligently extracts relevant information from the document and synthesizes concise, accurate, and context-aware answers, transforming how users interact with legal texts.

## Key Features
*   **Multi-Format Document Support**: Seamlessly process PDF, DOCX, and TXT legal documents.
*   **Intelligent Text Extraction**: Accurately extracts text content from various document types.
*   **Advanced Text Chunking**: Breaks down large documents into manageable, semantically relevant chunks for efficient processing.
*   **Google Cloud Vertex AI Integration**: Utilizes Vertex AI for robust embedding generation and powerful large language models (LLMs) for accurate answer synthesis.
*   **FAISS Vector Store**: Employs FAISS for efficient in-memory vector storage and retrieval, ensuring fast and relevant context lookup.
*   **Interactive Chat Interface**: Provides a user-friendly chat interface for asking questions and receiving AI-generated answers.
*   **Stateless Backend Design**: The backend is designed to be stateless, serializing the FAISS index and sending it to the frontend, enhancing scalability and robustness.
*   **Secure and Private Analysis**: Document processing occurs locally or within your secure Google Cloud environment, ensuring data privacy.

## Technology Stack

### Backend
*   **Python**: Core programming language.
*   **Flask**: Web framework for building the RESTful API.
*   **Flask-CORS**: Handles Cross-Origin Resource Sharing.
*   **python-dotenv**: Manages environment variables.
*   **Google Cloud AI Platform (Vertex AI)**: For embeddings (`textembedding-gecko@003`) and large language models (`gemini-1.0-pro`).
*   **LangChain**: Framework for building LLM applications, including text splitting and RAG chain orchestration.
*   **pypdf2**: For PDF document parsing.
*   **python-docx**: For DOCX document parsing.
*   **FAISS-CPU**: Efficient in-memory vector database for similarity search.
*   **pickle & base64**: For serializing and encoding the FAISS vector store.

### Frontend
*   **HTML5**: Structure of the web application.
*   **CSS3**: Styling and layout.
*   **JavaScript**: Client-side logic for UI interaction and API calls.

## Setup and Installation Guide

Legalify can be easily set up and run using Docker Compose, which is the recommended approach. A manual setup guide is also provided as an alternative.

### Recommended Setup (Docker Compose)

This method simplifies the setup process by containerizing the backend application and its dependencies.

#### Prerequisites
*   **Docker**: Ensure Docker Desktop (or Docker Engine) is installed and running on your system.
    *   [Install Docker](https://docs.docker.com/get-docker/)
*   **Docker Compose**: Docker Compose is usually included with Docker Desktop installations. Verify its presence.
    *   [Install Docker Compose](https://docs.docker.com/compose/install/)
*   **Google Cloud CLI**: Install and initialize the `gcloud` CLI. This is necessary for authenticating with Google Cloud services.
    *   [Install Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
    *   [Initialize gcloud CLI](https://cloud.google.com/sdk/docs/initializing)

#### Setup Steps

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-repo/legalify.git # Replace with your actual repo URL
    cd legalify
    ```

2.  **Authenticate with Google Cloud**:
    This command will open a browser window for you to log in to your Google account and grant permissions. This is crucial for the Vertex AI SDK to access your project from within the Docker container.
    ```bash
    gcloud auth application-default login
    ```

3.  **Set up your Google Cloud Project ID**:
    Create a file named `.env` in the **root** `legalify/` directory (the same directory as `docker-compose.yml`) with the following content. Replace `your-google-cloud-project-id` with your actual Google Cloud Project ID.
    ```
    GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
    ```

#### Running the Application

1.  **Launch the application using Docker Compose**:
    From the **root** `legalify/` directory, run the following command. The `--build` flag will build the Docker image if it hasn't been built yet or if changes have been made to the `Dockerfile` or `requirements.txt`.
    ```bash
    docker-compose up --build
    ```
    This will start the Flask backend server inside a Docker container.

2.  **Access the application**:
    *   The backend API will be accessible at `http://localhost:5000/`.
    *   Open the frontend in your web browser by navigating to the `frontend/index.html` file.
        ```bash
        # From the project root directory
        # For example, on Windows:
        start frontend\index.html
        # On macOS:
        open frontend/index.html
        # On Linux (using xdg-open, may vary):
        xdg-open frontend/index.html
        ```

    You can now upload documents and interact with the AI assistant!

### Manual Setup for Local Development (Alternative)

If you prefer not to use Docker, you can set up the backend manually.

#### Prerequisites
*   **Python 3.9+**: Ensure Python is installed on your system.
*   **Google Cloud CLI**: Install and initialize the `gcloud` CLI. This is necessary for authenticating with Google Cloud services.
    *   [Install Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
    *   [Initialize gcloud CLI](https://cloud.google.com/sdk/docs/initializing)

#### Backend Setup
1.  **Navigate to the `backend` directory**:
    ```bash
    cd backend
    ```

2.  **Create a Python virtual environment**:
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install required Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your Google Cloud Project ID**:
    Create a file named `.env` in the `backend` directory with the following content. Replace `your-google-cloud-project-id` with your actual Google Cloud Project ID.
    ```
    GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
    ```

6.  **Authenticate with Google Cloud**:
    This command will open a browser window for you to log in to your Google account and grant permissions. This is crucial for the Vertex AI SDK to access your project.
    ```bash
    gcloud auth application-default login
    ```

#### Running the Application

1.  **Start the Flask backend server**:
    Ensure your virtual environment is activated and you are in the `backend` directory.
    ```bash
    python app.py
    ```
    The server will typically run on `http://127.0.0.1:5000/`.

2.  **Open the frontend in your browser**:
    Navigate to the `frontend` directory in your project and open the `index.html` file in your web browser.
    ```bash
    # From the project root directory
    # For example, on Windows:
    start frontend\index.html
    # On macOS:
    open frontend/index.html
    # On Linux (using xdg-open, may vary):
    xdg-open frontend/index.html
    ```

    You can now upload documents and interact with the AI assistant!

