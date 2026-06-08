# FinRecon AI

A multi-source financial intelligence assistant powered by Retrieval-Augmented Generation (RAG). FinRecon AI enables natural language querying of financial reports and automatically detects discrepancies between transaction records. It is built to investigate how large language models can reduce manual effort in financial reconciliation workflows.

---

## What It Does

FinRecon AI combines two capabilities in a single RAG pipeline:

1. **Financial Report Q&A** — Ask natural language questions about a financial earnings report and receive grounded, context-aware answers backed by the source document
2. **Transaction Discrepancy Detection** — Automatically cross-references a bank ledger against internal records to identify amount mismatches, status conflicts, and missing transactions

### Example Queries

```bash
python query_data.py "What was TD's reported net income for Q2 2026?"
python query_data.py "What transactions have discrepancies between the bank ledger and internal records?"
python query_data.py "Which transactions are missing from internal records?"
python query_data.py "What were TD's adjusted earnings per share?"
```

### Example Output

```
Response: The transactions with discrepancies between the bank ledger and internal records are:
1. TXN009: Exists in the bank ledger (31000.0 CAD) but is missing from internal records.
2. TXN007: Amount mismatch. Bank ledger shows 4500.0 CAD, while internal records show 5200.0 CAD. Difference: 700.00 CAD.
3. TXN013: Amount mismatch. Bank ledger shows 6750.0 CAD, while internal records show 7100.0 CAD. Difference: 350.00 CAD.
Sources: ['discrepancy', 'discrepancy', 'discrepancy']
```

---

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   PDF Earnings      │     │   CSV Transaction   │
│   Report (18 pgs)   │     │   Records (x2)      │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────┐
│         Document Processing Pipeline            │
│  • PDF: Semantic chunking (1000 chars, 200      │
│    overlap) via LangChain                       │
│  • CSV: Row-level document conversion +         │
│    automated discrepancy detection              │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│         Azure OpenAI Embeddings                 │
│         (text-embedding-ada-002)                │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│         ChromaDB Vector Store                   │
│         (persistent local storage)              │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│         Semantic Similarity Search              │
│         (cosine similarity, k=3, 0.82+ scores) │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│         Azure OpenAI gpt-4o                     │
│         (answer generation)                     │
└─────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| LLM | Azure OpenAI (gpt-4o) |
| Embeddings | Azure OpenAI (text-embedding-ada-002) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| PDF Loader | LangChain PyPDFLoader |
| Data Processing | Pandas |
| Environment | python-dotenv |

---

## Project Structure

```
finrecon-ai/
├── data/
│   ├── TD-2026-q2-earnings-report.pdf   # TD Q2 2026 Earnings News Release
│   ├── bank_ledger.csv                  # Bank-side transaction records
│   └── internal_records.csv            # Internal system transaction records
├── chroma_db/                           # Persistent ChromaDB vector store
├── create_database.py                   # PDF ingestion and vector store creation
├── load_csv.py                          # CSV ingestion and discrepancy detection
├── query_data.py                        # Natural language query engine
├── requirements.txt                     # Project dependencies
├── .env.example                         # Environment variable template
└── .gitignore
```

---

## Setup

### Prerequisites
- Python 3.10+
- Azure OpenAI resource with the following deployments:
  - `gpt-4o` (chat completion)
  - `text-embedding-ada-002` (embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/ojaura/finrecon-ai.git
cd finrecon-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

### Running the Application

```bash
# Step 1: Ingest PDF and create vector store
python create_database.py

# Step 2: Load CSV transaction data and detect discrepancies
python load_csv.py

# Step 3: Query the system
python query_data.py "your question here"
```

---

## Planned Enhancements

- **Streamlit UI** — Web interface for uploading documents and querying without the command line
- **Multi-document support** — Ingest multiple financial reports across quarters for trend analysis
- **Expanded reconciliation logic** — Support for date range filtering, currency conversion, and threshold-based alerting
- **Model evaluation** — Automated response scoring using groundedness and relevance metrics

---

## Notes

- All amounts are in Canadian dollars (CAD) unless otherwise noted
- Built with Azure OpenAI to align with enterprise production standards used in regulated financial environments
- The PDF used is TD Bank Group's publicly available Q2 2026 Earnings News Release
