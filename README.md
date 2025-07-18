‼ ⚠ This readme file might be outdated if you see this message. It will be updated once the whole project is ready.
# EduGen

EduGen is a microservices-based application aimed at supporting the learning process. The project consists of two main Python microservices, leveraging RAG (Retrieval-Augmented Generation) with LLM (Large Language Model) to generate educational content such as quizzes, learning plans, summaries, and cheat sheets.

## Table of Contents

- [Project Description](#project-description)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Microservices](#running-the-microservices)
- [Testing](#testing)
- [Technologies Used](#technologies-used)
- [Author](#author)

## Project Description

EduGen is an application that assists the learning process by generating personalized educational content. The project consists of two microservices:
- `pdf_processing_service` - Responsible for processing PDF documents, generating embeddings, and indexing them.
- `rag_llm_service` - Integrates with the LLM model to generate quizzes, learning plans, summaries, and cheat sheets based on the provided material.

## Architecture

The project is built based on microservices that communicate via REST APIs. Each microservice has a dedicated responsibility:
- **`pdf_processing_service`**: Handles PDF documents, extracts content, generates embeddings, and stores them in a vector database (e.g., Pinecone).
- **`rag_llm_service`**: Uses LLM to generate educational content based on the provided data.

The microservices will later be integrated via an API Gateway, such as **Ocelot**, and the entire system will be scaled in a microservices architecture.

## Requirements

- Python 3.8+
- Pipenv / venv for virtual environment management
- Libraries listed in `requirements.txt` for each microservice
- API key for OpenAI (for LLM integration)
- Pinecone account (optional, for storing embeddings)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/kottoization/EduGen.git
cd EduGen
```

### Step 2: Install Dependencies

#### `pdf_processing_service`

```bash
cd pdf_processing_service
python -m venv venv
source venv/bin/activate  # For Unix systems
# .\venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

#### `rag_llm_service`

```bash
cd ../rag_llm_service
python -m venv venv
source venv/bin/activate  # For Unix systems
# .\venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

## Running the Microservices

### `pdf_processing_service`

Run the FastAPI server:

```bash
cd pdf_processing_service
uvicorn api.main:app --reload --port 8001
```

### `rag_llm_service`

Run the FastAPI server:

```bash
cd rag_llm_service
uvicorn api.main:app --reload --port 8002
```

### `frontend_service`

Launch the Gradio web interface:

```bash
cd frontend_service
python main.py
```

Each microservice will be available on the respective port (e.g., `http://localhost:8001` for `pdf_processing_service`).

## Testing

To test the modules, use `pytest`.

### Example of Running Tests:

#### `pdf_processing_service`

```bash
cd pdf_processing_service
pytest tests/
```

#### `rag_llm_service`

```bash
cd rag_llm_service
pytest tests/
```

## Technologies Used

- **Python**: Programming language used to develop the microservices.
- **FastAPI**: Framework for building REST APIs.
- **OpenAI API**: For generating content using LLM.
- **Sentence Transformers**: For generating text embeddings.
- **Pinecone/Faiss**: For storing and retrieving embeddings.
- **Redis**: For caching responses.
- **Pytest**: Tool for testing the code.

## Project Files and Structure

- **`pdf_processing_service/`**: PDF processing and embedding generation microservice.
- **`rag_llm_service/`**: Educational content generation microservice.
- **`frontend_service/`**: Gradio-based chat frontend.
- **`README.md`**: Project documentation.
- **`.gitignore`**: File ignoring temporary files, dependencies, and sensitive data.

### Auto-answer

Whenever an input string ends with a question mark, the system treats it as an
on-demand question for the agent. The agent's reply is printed and the prompt is
repeated. This behaviour is used throughout the CLI, including during quiz
answers and flashcard review.

## Author

- **Mateusz Mulka** - [GitHub](https://github.com/kottoization)

---

### Notes

- Make sure the `.env` file is correctly configured before running the microservices.
- Adjust configuration parameters in `config.yaml` files and other settings as needed for your project.
- Before deploying to production, test the scalability, performance, and security of each microservice.
