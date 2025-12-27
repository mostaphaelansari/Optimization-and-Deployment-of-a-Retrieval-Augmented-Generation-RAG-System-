# 🤖 RAG System — Retrieval Augmented Generation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**A production-ready RAG pipeline for intelligent question-answering over PDF documents**

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Architecture](#-architecture) •
[Evaluation](#-evaluation) •
[Team](#-team)

</div>

---

## 📋 Overview

This project implements a complete **Retrieval Augmented Generation (RAG)** system that enables intelligent question-answering over PDF documents. The system combines semantic search with local Large Language Models to provide accurate, context-aware responses.

### 📄 Research Papers Used

| Paper | Authors | Year | Focus |
|-------|---------|------|-------|
| **Attention Is All You Need** | Vaswani et al. | 2017 | Transformer Architecture |
| **BERT: Pre-training of Deep Bidirectional Transformers** | Devlin et al. | 2018 | Bidirectional Language Models |
| **Language Models are Few-Shot Learners (GPT-3)** | Brown et al. | 2020 | Few-Shot Learning |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 📚 Document Processing
- PDF loading and parsing
- Intelligent text chunking (1000 chars)
- Metadata preservation
- Recursive text splitting

</td>
<td width="50%">

### 🔍 Semantic Search
- Vector embeddings (MiniLM-L6)
- ChromaDB vector store
- Similarity scoring
- Top-K retrieval

</td>
</tr>
<tr>
<td width="50%">

### 🤖 LLM Integration
- Local inference via Ollama
- Qwen 2.5 (1.5B) model
- Custom prompt templates
- Context-aware responses

</td>
<td width="50%">

### 💬 Interactive Interfaces
- Beautiful CLI with Rich
- Streamlit Web UI
- Conversation history
- Source citations

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            RAG PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   📄 PDFs                                                               │
│      │                                                                  │
│      ▼                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   Loading   │───▶│  Chunking   │───▶│ Embeddings  │                │
│   │  (PyPDF)    │    │  (1000ch)   │    │  (MiniLM)   │                │
│   └─────────────┘    └─────────────┘    └─────────────┘                │
│                                                │                        │
│                                                ▼                        │
│                                         ┌─────────────┐                │
│                                         │  ChromaDB   │                │
│                                         │ Vector Store│                │
│                                         └─────────────┘                │
│                                                │                        │
│   ┌─────────────┐    ┌─────────────┐          │                        │
│   │   Answer    │◀───│   Ollama    │◀─────────┘                        │
│   │             │    │  (Qwen2.5)  │                                   │
│   └─────────────┘    └─────────────┘                                   │
│         │                   ▲                                          │
│         │            ┌─────────────┐    ┌─────────────┐                │
│         │            │   Prompt    │◀───│  Retriever  │                │
│         │            │  Template   │    │   (Top-K)   │                │
│         │            └─────────────┘    └─────────────┘                │
│         ▼                                      ▲                        │
│   ┌─────────────┐                              │                        │
│   │  User Query │──────────────────────────────┘                        │
│   └─────────────┘                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RAG-Project/
│
├── 📄 cli.py                      # Command-line interface
├── 📄 app.py                      # Streamlit web application
├── ⚙️ config.yaml                 # System configuration
├── 📋 requirements.txt            # Python dependencies
├── 📝 template.py                 # Prompt templates
├── 📖 README.md                   # Project documentation
│
├── 📂 data/
│   ├── 1706.03762v7.pdf          # Attention Is All You Need
│   ├── 1810.04805v2.pdf          # BERT paper
│   ├── 2005.14165v4.pdf          # GPT-3 paper
│   └── evaluation_dataset.json   # Test questions & ground truths
│
├── 📂 src/
│   ├── __init__.py
│   ├── document_indexer.py       # Q1: Document loading & chunking
│   ├── vector_store.py           # Q1: ChromaDB vector storage
│   ├── document_retriever.py     # Q2: Semantic retrieval
│   ├── llm_qa_system.py          # Q3: LLM question-answering
│   ├── evaluator.py              # Q4: Evaluation metrics
│   ├── chatbot.py                # Q5: Conversational chatbot
│   │
│   └── 📂 utils/
│       ├── __init__.py
│       ├── config_loader.py      # Configuration management
│       ├── logger.py             # Logging utilities
│       └── metrics.py            # Evaluation metrics
│
└── 📂 vector_store/              # Persisted embeddings (gitignored)
```

---

## 🚀 Installation

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| Ollama | Latest | Local LLM |
| CUDA | 11.8+ | GPU acceleration (optional) |

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/RAG-Project.git
cd RAG-Project
```

### Step 2: Create Virtual Environment

```bash
# Using Conda (recommended)
conda create -n rag python=3.10
conda activate rag

# Or using venv
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support (optional, for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Setup Ollama

```bash
# Download Ollama from https://ollama.com/download

# Pull the LLM model
ollama pull qwen2.5:1.5b

# Verify installation
ollama list
```

---

## 💻 Usage

### Quick Start

```bash
# 1️⃣ Index your documents
python cli.py index data/ -d

# 2️⃣ Ask a question
python cli.py ask "What is the Transformer architecture?" -s

# 3️⃣ Start the web interface
streamlit run app.py
```

### CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `index` | Index PDF documents | `python cli.py index data/ -d` |
| `search` | Semantic search | `python cli.py search "attention mechanism"` |
| `ask` | Ask a question | `python cli.py ask "What is BERT?" -s` |
| `chat` | Interactive chatbot | `python cli.py chat` |
| `evaluate` | Run evaluation | `python cli.py evaluate -o results.json` |
| `stats` | Vector store info | `python cli.py stats` |
| `models` | List Ollama models | `python cli.py models` |
| `config` | Show configuration | `python cli.py config` |
| `web` | Launch Streamlit | `python cli.py web` |

### Web Interface

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**Features:**
- 💬 **Chat**: Interactive conversation with history
- ❓ **Q&A**: Single questions with source citations
- 🔍 **Search**: Semantic document search

---

## ⚙️ Configuration

All settings are centralized in `config.yaml`:

```yaml
# Document Processing
document_processing:
  chunk_size: 1000          # Characters per chunk
  chunk_overlap: 200        # Overlap between chunks
  split_method: "recursive" # Splitting strategy

# Embeddings
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"            # Use GPU if available

# LLM (Ollama)
llm:
  model_name: "qwen2.5:1.5b"
  base_url: "http://localhost:11434"
  temperature: 0.7

# Retrieval
retrieval:
  top_k: 5                  # Number of chunks to retrieve
  score_threshold: 0.3      # Minimum similarity score
```

---

## 📊 Evaluation

### Run Evaluation

```bash
python cli.py evaluate -o results.json
```

### Metrics

#### Retrieval Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **Precision@5** | 0.98 | Relevant documents in top 5 |
| **Recall@5** | 0.90 | Fraction of relevant docs retrieved |
| **MRR** | 1.00 | Mean Reciprocal Rank |
| **Hit Rate@5** | 1.00 | Success rate for finding relevant docs |

#### Answer Quality

| Metric | Score | Description |
|--------|-------|-------------|
| **Answer Relevance** | 0.77 | How well answer addresses question |
| **Faithfulness** | 0.36 | Grounding in retrieved context |
| **Word Overlap F1** | 0.23 | Lexical similarity to ground truth |

---

## 🔧 Technical Choices

### Why These Technologies?

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Embedding Model** | `all-MiniLM-L6-v2` | Lightweight (80MB), fast, good semantic quality |
| **Vector Store** | ChromaDB | Easy setup, persistent storage, LangChain integration |
| **LLM** | Qwen 2.5 (1.5B) | Local inference, no API costs, fast (~1s response) |
| **Text Splitter** | RecursiveCharacterTextSplitter | Respects document structure, configurable |
| **Chunk Size** | 1000 characters | Balance between context richness and precision |

### Alternatives Considered

| Component | Alternative | Why Not Chosen |
|-----------|-------------|----------------|
| Embeddings | `all-mpnet-base-v2` | Better quality but slower |
| Vector Store | FAISS | Faster but no built-in persistence |
| LLM | Mistral-7B | Better quality but requires more VRAM |

---

## 📈 Sample Output

```
╭─────────────────────── 💡 Answer ───────────────────────╮
│                                                         │
│  The Transformer is a neural network architecture       │
│  designed to process sequences of data. It consists     │
│  of stacked self-attention mechanisms followed by       │
│  point-wise, fully connected layers for both encoder    │
│  and decoder. Its key components include multi-head     │
│  self-attention and position-wise feedforward networks. │
│                                                         │
╰─────────────────────────────────────────────────────────╯

               📚 Sources
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━┓
┃ Document              ┃ Page ┃ Score  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━┩
│ 1706.03762v7.pdf      │ 2    │ 0.5064 │
│ 1810.04805v2.pdf      │ 2    │ 0.4662 │
└───────────────────────┴──────┴────────┘
```

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| **Indexing Speed** | ~3 seconds for 3 PDFs |
| **Search Latency** | ~50ms per query |
| **Answer Generation** | ~1-2 seconds |
| **Memory Usage** | ~2GB VRAM |

---

## 👥 Team

| GitHub | Contributor |
|--------|-------------|
| [@mostaphaelansari](https://github.com/mostaphaelansari) | Mostapha El Ansari |
| [@elkhilyass](https://github.com/elkhilyass) | Ilyass El KHAZANE |
| [@akiraaymane](https://github.com/akiraaymane) | Aymane Dhimen |
| [@mendyvincent](https://github.com/mendyvincent) | Vincent Mendy |

| Marouane Rbib|
---

## 📚 References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Research Papers

1. Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. Devlin, J., et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
3. Brown, T., et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

---

## 📄 License

This project is developed for educational purposes as part of the RAG TP assignment.

---

<div align="center">

**Built with ❤️ using LangChain, ChromaDB, Ollama & Streamlit**

</div>
