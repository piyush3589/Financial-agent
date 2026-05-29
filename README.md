# Financial Document Intelligence Agent

A production-grade AI agent that extracts, validates, and analyzes financial documents using LangGraph, RAG, and OCR.

## Features

- **Document Processing**: Extract text from PDFs and images via OCR (Tesseract)
- **LangGraph Agent**: Multi-step orchestration (extract → validate → detect anomalies → audit)
- **Financial Validation**: Check extracted data against financial rules
- **Anomaly Detection**: Flag suspicious patterns and missing fields
- **Audit Trail**: SQLite database logging for compliance
- **REST API**: FastAPI endpoints for document processing
- **Production-Ready**: Docker deployment, error handling, structured responses

## Architecture

```
User Upload (PDF/Image)
    ↓
[OCR: Tesseract] ← Extracts text
    ↓
[LangGraph Agent]
    ├→ Extract Node: LLM-based data extraction
    ├→ Validate Node: Rule-based validation
    ├→ Anomaly Node: Pattern detection
    └→ Audit Node: Database logging
    ↓
JSON Response (extracted data, validation, anomalies, confidence)
```

## Setup (Windows)

### Prerequisites
- Python 3.11+
- Tesseract OCR (Windows installer)
- Git

### Step 1: Install Tesseract
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

Default Windows path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### Step 2: Clone and setup
```bash
git clone <your-repo-url>
cd financial_agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure environment
```bash
copy .env.example .env
# Edit .env and add your Groq API key
```

Get your Groq API key: https://console.groq.com (free tier available)

### Step 4: Run locally
```bash
python main.py
```

Visit: http://localhost:8000/docs

### Step 5: Test
Upload a PDF or image via the `/process-document` endpoint.

## API Endpoints

### GET /health
Health check.
```bash
curl http://localhost:8000/health
```

### POST /process-document
Process a financial document.
```bash
curl -X POST "http://localhost:8000/process-document" \
  -F "file=@invoice.pdf"
```

Response:
```json
{
  "log_id": "log_2024-01-15T10:30:45.123456",
  "extracted_data": {
    "dates": ["2024-01-15"],
    "amounts": [1500.00],
    "entities": ["Acme Corp"],
    "transaction_id": "INV-2024-001",
    "line_items": [...],
    "document_type": "invoice"
  },
  "validation_results": {
    "valid": true,
    "issues": [],
    "completeness_score": 0.95,
    "confidence": 0.95
  },
  "anomalies": [],
  "confidence": 0.95,
  "status": "processed"
}
```

### GET /audit-logs
Retrieve recent processing logs.
```bash
curl "http://localhost:8000/audit-logs?limit=10"
```

## Database

SQLite database `financial_agent.db` stores:
- Log ID
- Timestamp
- Document summary
- Extracted data (JSON)
- Validation results (JSON)
- Confidence score

Query example:
```bash
sqlite3 financial_agent.db "SELECT id, timestamp, confidence_score FROM audit_log LIMIT 5;"
```

## Deployment (Docker)

### Build
```bash
docker build -t financial-agent .
```

### Run locally
```bash
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  financial-agent
```

### Deploy to Railway (free)
1. Push to GitHub
2. Connect repo to Railway
3. Add environment variable: `GROQ_API_KEY`
4. Deploy

## Next Steps (Week 2+)

### Add RAG Pipeline
- Integrate Pinecone vector DB
- Store financial document patterns as embeddings
- Use retrieval to improve extraction accuracy

### Add LangGraph Memory
- Multi-turn conversations about documents
- Context persistence across requests

### Expand Validation Rules
- GAAP-specific checks
- Industry-specific anomaly detection

## Troubleshooting

**Tesseract not found:**
```
ModuleNotFoundError: No module named 'pytesseract'
```
→ Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki

**Groq API key invalid:**
→ Check .env file and regenerate key from https://console.groq.com

**PDF extraction fails:**
→ Try with a simpler PDF first; complex PDFs may need alternative OCR

## Project Structure
```
financial_agent/
├── main.py           # FastAPI app
├── agent.py          # LangGraph agent + tools
├── requirements.txt  # Dependencies
├── .env              # API keys (git-ignored)
├── .env.example      # Template
├── Dockerfile        # Docker config
├── README.md         # This file
└── financial_agent.db # SQLite (auto-created)
```

## Tech Stack

- **LangGraph**: Agent orchestration
- **Groq API**: Free LLM (mixtral-8x7b)
- **Tesseract**: OCR (images & PDFs)
- **FastAPI**: REST API
- **SQLAlchemy**: Database ORM (ready for expansion)
- **Docker**: Containerization

## License

MIT

---

**Built for portfolio/interview preparation. Real financial validation requires domain expertise and regulatory compliance review.**
