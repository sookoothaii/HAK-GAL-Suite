# HAK-GAL Suite: A Hybrid AI Framework for Verifiable Knowledge and Reasoning

## Overview

**HAK-GAL (Hybrid Assertion Knowledge & Grounded Assertion Logic)** is an advanced, modular framework that integrates the natural language processing capabilities of Large Language Models (LLMs) with the formal rigor of symbolic logic and automated theorem proving.
The primary objective is to enable robust reasoning over unstructured data and produce **verifiable, explainable, and trustworthy outputs**, thus minimizing the risk of factual inaccuracies ("hallucinations") inherent in generative models.

This repository contains the complete source code for the HAK-GAL engine, a reference web application, and all associated documentation.

---

## Core Features

- **Hybrid Architecture**: Combines neural network-based language models for semantic understanding with a symbolic core for logical validation and inference.
- **Logic Firewall**: Enforces a strict, custom-defined grammar (`HAKGAL_GRAMMAR`) and uses a Lark-based parser to ensure formal syntactic correctness.
- **Provable Reasoning**: Utilizes a Z3 SMT solver to formally prove or disprove logical statements, ensuring logical soundness, not just probability.
- **Retrieval-Augmented Generation (RAG)**: Integrates external knowledge sources (e.g., academic papers, technical documents) to ground its reasoning in verifiable data.
- **Interactive Clarification**: Features a dialogue system that resolves ambiguities, incomplete queries, or contradictions through user interaction.
- **Performance Optimization**: Employs multi-layered caching (prompt and proof caches) to reduce latency and computational overhead.
- **Modular Design**: Clear separation between UI, backend API, and core logic ensures maintainability, scalability, and extensibility.

---

## Project Structure

The repository is organized into the following main directories:

```
/
├── frontend/     # Contains the Vite/React/TypeScript web application
├── backend/      # Contains the Python-based core logic, API, and Z3 integration
└── docs/         # All project documentation
```

---

## Technology Stack

- **Frontend**:
  - React
  - Vite
  - TypeScript
  - Tailwind CSS

- **Backend**:
  - Python 3.9+
  - Z3 SMT Solver (for theorem proving)
  - Lark Parser (for grammatical enforcement)
  - Flask (for API)

- **Development & Deployment**:
  - Docker & Docker Compose
  - GitHub Actions for CI/CD

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js (v18+) & a package manager (npm, yarn, or pnpm)
- Docker & Docker Compose (recommended for easiest setup)
- API keys for your LLM providers (e.g., OpenAI, Anthropic).

### Installation & Execution

#### 1. Clone the Repository

```bash
git clone https://github.com/sookoothaii/HAK-GAL-Suite.git
cd HAK-GAL-Suite
```

#### 2. Running the Backend

The backend is built with Python and Flask. Create a `.env` file in the root directory for your API keys.

```bash
# Create and populate the .env file in the root directory
# Example:
# OPENAI_API_KEY="sk-..."
# ANTHROPIC_API_KEY="..."

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r docs/requirements.txt

# Run the development server
python api.py
```
The backend API will typically be available at `http://localhost:5001` (as configured in the api.py file).

#### 3. Running the Frontend

The frontend is a modern Vite application.

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The application will then be available at `http://localhost:5173`.

---

## Contribution

Interested in contributing? Please read our [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on how to get started, our code of conduct, and the process for submitting pull requests.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
