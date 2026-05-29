from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image
import io
import pdf2image
from agent import process_document
from pydantic import BaseModel
import json
from typing import Optional

app = FastAPI(title="Financial Document Intelligence Agent")

# Configure Tesseract path for Windows
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============ MODELS ============
class ProcessDocumentResponse(BaseModel):
    log_id: str
    extracted_data: dict
    validation_results: dict
    anomalies: dict
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str


# ============ UTILITIES ============
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using OCR."""
    try:
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=200)
        text = ""
        for image in images[:5]:  # Limit to first 5 pages
            text += pytesseract.image_to_string(image) + "\n"
        return text
    except Exception as e:
        return f"PDF extraction failed: {str(e)}"

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Image extraction failed: {str(e)}"


# ============ ENDPOINTS ============
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Financial Document Intelligence Agent is running"
    }

@app.post("/process-document", response_model=ProcessDocumentResponse)
async def process_document_endpoint(file: UploadFile = File(...)):
    """
    Process a financial document (PDF or image).
    Returns extracted data, validation results, and anomalies.
    """
    try:
        # Read file
        contents = await file.read()
        file_ext = file.filename.split(".")[-1].lower()
        
        # Extract text based on file type
        if file_ext == "pdf":
            text = extract_text_from_pdf(contents)
        elif file_ext in ["jpg", "jpeg", "png", "gif"]:
            text = extract_text_from_image(contents)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or image.")
        
        if not text or "extraction failed" in text.lower():
            raise HTTPException(status_code=400, detail=f"Could not extract text: {text}")
        
        # Run agent
        print(f"\n[API] Processing document: {file.filename}")
        print(f"[API] Extracted text length: {len(text)} chars")
        
        result = process_document(text)
        
        return ProcessDocumentResponse(
            log_id=result["log_id"],
            extracted_data=result["extracted_data"],
            validation_results=result["validation_results"],
            anomalies=result["anomalies"],
            status=result["status"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/audit-logs")
async def get_audit_logs(limit: int = 10):
    """Retrieve recent audit logs."""
    import sqlite3
    try:
        conn = sqlite3.connect("financial_agent.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, confidence_score FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        logs = cursor.fetchall()
        conn.close()
        
        return {
            "count": len(logs),
            "logs": [
                {"id": log[0], "timestamp": log[1], "confidence": log[2]}
                for log in logs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")


# ============ ROOT ============
@app.get("/")
async def root():
    return {
        "service": "Financial Document Intelligence Agent",
        "endpoints": {
            "GET /health": "Health check",
            "POST /process-document": "Process PDF or image (form-data, key='file')",
            "GET /audit-logs": "View recent processing logs"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Add this new endpoint before the root endpoint

@app.post("/process-document-with-rag")
async def process_document_with_rag_endpoint(file: UploadFile = File(...), use_rag: bool = True):
    """
    Process a financial document with RAG enhancement.
    RAG retrieves similar documents to improve accuracy.
    """
    try:
        from agent import process_document_with_rag
        
        # Read file
        contents = await file.read()
        file_ext = file.filename.split(".")[-1].lower()
        
        # Extract text
        if file_ext == "pdf":
            text = extract_text_from_pdf(contents)
        elif file_ext in ["jpg", "jpeg", "png", "gif"]:
            text = extract_text_from_image(contents)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        
        if not text or "extraction failed" in text.lower():
            raise HTTPException(status_code=400, detail=f"Could not extract text: {text}")
        
        print(f"\n[API] Processing document with RAG: {file.filename}")
        
        result = process_document_with_rag(text, use_rag=use_rag)
        
        return {
            "log_id": result["log_id"],
            "extracted_data": result["extracted_data"],
            "validation_results": result["validation_results"],
            "anomalies": result["anomalies"],
            "status": result["status"],
            "rag_enabled": result["rag_enabled"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
