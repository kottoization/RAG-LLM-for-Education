# EduGen

EduGen is a collection of utilities for generating learning materials with the help of Large Language Models.  
The repository contains a command line interface together with a simple Gradio based frontend.  
It can generate quizzes, flashcards, summaries and cheat sheets.  
Optionally the tools can use RAG (Retrieval Augmented Generation) on your local documents.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Gradio Frontend](#gradio-frontend)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Author](#author)

## Requirements
- Python 3.10+
- An OpenAI API key set in the `OPENAI_API_KEY` environment variable or in a `.env` file
- Packages from `requirements.txt`

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/kottoization/EduGen.git
cd EduGen
pip install -r requirements.txt
```

Create a `.env` file containing your OpenAI key:

```bash
OPENAI_API_KEY=your-key-here
```

## Usage

### CLI
Run the interactive console menu (optional):

```bash
python main.py --cli
```

The menu lets you chat with the assistant, generate quizzes, summaries, flashcards and learning plans.
If RAG documents are indexed (see `RAGModule`), some tools can enrich answers with your own files.

### Gradio Frontend
The application now launches the Gradio interface by default:

```bash
python main.py
```

You can also run the standalone frontend directly with:

```bash
python frontend_service/main.py
```

Both entry points import the same agent used by the optional CLI so you get identical behaviour in the browser.

## Project Structure
- `AgentModule/` – creation of the LangChain agent and reusable tools
- `RAGModule/` – utilities for loading documents and building a Chroma vector store
- `QuizModule/`, `FlashcardsModule/`, `LearningPlanModule/`, `SummaryModule/`, `CheatSheetModule/` – content generation helpers. The cheat sheet tool now
  follows a structured three-section layout inspired by the [RStudio cheatsheet guidelines](https://github.com/rstudio/cheatsheets/blob/main/.github/CONTRIBUTING.md) for clearer review notes.
- `frontend_service/` – Gradio based chat interface
- `data/` – example data and vector store persistence

## Testing
No automated tests are provided yet, but you can run `pytest` to verify that none are failing:

```bash
pytest
```

## Author
Mateusz Mulka – [kottoization](https://github.com/kottoization)
