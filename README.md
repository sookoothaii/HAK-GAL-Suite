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

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js (with npm, yarn, or pnpm)
- Docker & Docker Compose (recommended)
- API keys for your LLM providers (e.g., OpenAI, Anthropic)

### Installation & Execution

#### Clone the Repository

```bash
git clone https://github.com/sookoothaii/HAK-GAL-Suite.git
cd HAK-GAL-Suite
