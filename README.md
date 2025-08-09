# RAG LLM for education | EduGen

RAG LLM system for personalized education, called EduGen is a modular learning-material generation toolkit built around Large
Language Models and retrieval‑augmented generation techniques.  It exposes a
command‑line interface and a Gradio web application that together provide a
suite of interactive study aids:

- ReAct-based educational agent with tool augmentation.
- Quiz generation and performance analysis.
- Automated flashcard creation and review.
- Personalized learning plan generator.
- Detailed study summaries.
- Exam-style cheat sheets.

The system is multilingual and can translate or detect language, and many
components accept optional retrievers enabling retrieval‑augmented generation
when external knowledge bases are available.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [Gradio Frontend](#gradio-frontend)
- [Modules](#modules)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Author](#author)

## Features
- ReAct agent built with LangChain using OpenAI chat models and tools for
  Wikipedia lookup, dictionary definitions, math evaluation and date utilities.
- Automatic language detection and translation for inputs and outputs.
- Quiz generation with topic expansion and multiple‑choice questions, plus
  score‑based learning plan creation.
- Flashcard extraction from quiz text or LLM prompts with CLI review and JSON
  persistence.
- Study summary and cheat sheet generators with optional retrieval‑augmented
  context.
- Gradio interface combining chat, quiz, learning plan, flashcards, summary and
  cheat sheet workflows.

## Architecture
EduGen is organized as independent modules that share a common toolset and
language handling layer. Most modules accept an optional `retriever` object
implementing LangChain’s document retrieval interface to enable
retrieval‑augmented generation (RAG). The primary components are:

- **AgentModule** – creates a LangChain `AgentExecutor` configured with ReAct
  prompting and custom tools. `run_agent` wraps execution and falls back to the
  raw LLM when tool calls fail.
- **QuizModule** – uses prompt templates and parallel `Runnable` chains to
  produce quiz topics and questions. Results can feed into the
  `LearningPlanModule`.
- **FlashcardsModule** – parses quiz text or prompts an LLM to generate Q/A
  pairs, supports CLI review and saves or loads decks from JSON files.
- **LearningPlanModule** – analyses quiz scores, prioritises topics, suggests
  study materials via the LLM and persists plans to disk.
- **SummaryModule** – builds multi‑section study guides using a detailed
  lecturer‑style prompt.
- **CheatSheetModule** – outputs a compact one‑page reference with key terms,
  formulas and facts.
- **frontend_service** – Gradio application exposing all capabilities through
  tabs; uses a shared agent and `LanguageHandler` to ensure responses in the
  selected language.
- **tools** – utility functions including language detection/translation, quiz
  prompt builders, auto‑answer heuristics and reusable agent tools.

## Tech Stack
EduGen is implemented primarily in **Python 3.10+** and builds on the
following libraries and techniques:

- **LangChain** for chaining LLM calls, building ReAct agents and composing
  retrievers.
- **OpenAI GPT‑3.5/4** models for natural language generation.
- **Gradio** for the optional web user interface.
- **Chroma** as the default vector store for document retrieval.
- **Retrieval‑Augmented Generation (RAG)** with optional MultiQuery and
  Maximal Marginal Relevance search to improve context.
- **pytest** for automated testing.

The project also includes helper utilities for language detection/translation
and document ingestion into the vector store.

## Requirements
- Python 3.10 or newer
- OpenAI API key exposed via the `OPENAI_API_KEY` environment variable or a
  local `.env` file
- Dependencies from `requirements.txt`

## Installation
```bash
git clone https://github.com/kottoization/EduGen.git
cd EduGen
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI key:

```bash
OPENAI_API_KEY=your-key-here
```

### Environment variables
Several knobs control RAG behaviour:

- `RAG_K` – number of documents retrieved (default: 4)
- `RAG_USE_MMR` – enable Maximal Marginal Relevance search (default: true)
- `RAG_USE_MULTIQUERY` – expand queries with an LLM (default: true)

## Usage

### CLI
```bash
python main.py --cli
```
The menu allows chatting with the agent, generating quizzes, learning plans,
flashcards, summaries and cheat sheets.

### Gradio Frontend
```bash
python main.py
```
This launches the web UI. The interface can also be run directly:

```bash
python frontend_service/main.py
```

## Modules
Each module can be imported and used independently. Typical entry points
include:

- `AgentModule.create_agent()` – construct a ReAct agent with default tools.
- `QuizModule.prepare_quiz_questions(subject, language, retriever=None)` –
  generate structured question dictionaries.
- `FlashcardsModule.FlashcardSet.generate_from_prompt(topic_prompt, language)` –
  create flashcards via LLM and optional retrieval.
- `LearningPlanModule.LearningPlan.generate_plan()` – build a prioritised
  schedule from quiz results.
- `SummaryModule.StudySummaryGenerator.generate_summary(text, language)` –
  produce detailed study notes.
- `CheatSheetModule.CheatSheetGenerator.generate_cheatsheet(text, language)` –
  return concise exam revision sheets.

## Testing
Automated tests validate core utilities. Run all tests with:

```bash
pytest
```

## Project Structure
```
AgentModule/            # LangChain agent definition and tools
CheatSheetModule/       # Cheat sheet generator
FlashcardsModule/       # Flashcard classes and generators
LearningPlanModule/     # Learning plan creation based on quiz results or goals
QuizModule/             # Quiz question generation and interactive runner
SummaryModule/          # Study summary generator
frontend_service/       # Gradio web interface
tools/                  # Shared utilities (language, prompts, agent tools)
data/                   # Sample data, saved flashcards and plans
tests/                  # Unit tests
```

## Author
Mateusz Mulka – [kottoization](https://github.com/kottoization)

