import json
import os
from datetime import datetime
import sqlite3
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import pytesseract

# Configure paths
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['PATH'] += r';C:\Program Files\poppler-24.08.0\Library\bin'

# ============ DATABASE SETUP ============
def init_db():
    conn = sqlite3.connect("financial_agent.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            document_summary TEXT,
            extracted_data TEXT,
            validation_results TEXT,
            confidence_score REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_db(doc_summary, extracted, validation, confidence):
    conn = sqlite3.connect("financial_agent.db")
    cursor = conn.cursor()
    log_id = f"log_{datetime.now().isoformat()}"
    cursor.execute("""
        INSERT INTO audit_log (id, timestamp, document_summary, extracted_data, validation_results, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (log_id, datetime.now().isoformat(), doc_summary, json.dumps(extracted), json.dumps(validation), confidence))
    conn.commit()
    conn.close()
    return log_id

init_db()

# ============ AGENT FUNCTIONS ============
def extract_financial_data(text: str) -> dict:
    """Extract structured financial data from document text."""
    extraction_prompt = f"""
    Extract financial data from this document text. Return ONLY valid JSON, no markdown.
    
    Look for:
    - dates (any format)
    - amounts (numbers with currency symbols or decimal points)
    - vendor/account names
    - invoice/transaction IDs
    - line items
    
    Document text:
    {text[:2000]}
    
    Return ONLY this JSON structure, nothing else:
    {{
        "dates": [],
        "amounts": [],
        "entities": [],
        "transaction_id": null,
        "line_items": [],
        "document_type": "unknown"
    }}
    """
    
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    
    try:
        result = json.loads(response.content)
    except:
        result = {
            "dates": [],
            "amounts": [],
            "entities": [],
            "transaction_id": None,
            "line_items": [],
            "document_type": "unparseable"
        }
    
    return result

def validate_financial_data(extracted_data: dict) -> dict:
    """Validate extracted financial data against common rules."""
    issues = []
    
    # Check for required fields
    if not extracted_data.get("amounts"):
        issues.append("No amounts found")
    if not extracted_data.get("dates"):
        issues.append("No dates found")
    if not extracted_data.get("entities"):
        issues.append("No entities (vendor/account) found")
    
    # Check amount reasonableness
    for amount in extracted_data.get("amounts", []):
        try:
            amt = float(amount) if isinstance(amount, str) else amount
            if amt > 10000000:
                issues.append(f"Amount {amt} seems unusually large")
            if amt < 0:
                issues.append(f"Negative amount {amt} flagged")
        except:
            pass
    
    # Confidence based on data completeness
    completeness = sum([
        bool(extracted_data.get("dates")),
        bool(extracted_data.get("amounts")),
        bool(extracted_data.get("entities")),
        bool(extracted_data.get("transaction_id"))
    ]) / 4
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "completeness_score": round(completeness, 2),
        "confidence": round(max(0.5, completeness), 2)
    }

def detect_anomalies(extracted_data: dict, validation_results: dict) -> dict:
    """Detect anomalies in financial data."""
    anomalies = []
    
    # Add validation issues as anomalies
    anomalies.extend(validation_results.get("issues", []))
    
    # Check for missing transaction ID
    if extracted_data.get("document_type") == "invoice" and not extracted_data.get("transaction_id"):
        anomalies.append("Invoice without transaction ID")
    
    # Check line item count
    if extracted_data.get("line_items"):
        if len(extracted_data["line_items"]) > 100:
            anomalies.append("Unusually high number of line items (>100)")
    
    return {
        "anomalies_detected": len(anomalies) > 0,
        "anomaly_list": anomalies,
        "severity": "high" if len(anomalies) > 3 else "medium" if len(anomalies) > 0 else "low"
    }

def process_document(document_text: str) -> dict:
    """Main orchestration function - runs the full agent workflow."""
    print("\n[AGENT] Starting document processing...")
    
    # Step 1: Extract
    print("[AGENT] Step 1: Extracting financial data...")
    extracted = extract_financial_data(document_text)
    print(f"[AGENT] Extracted: {json.dumps(extracted, indent=2)}")
    
    # Step 2: Validate
    print("[AGENT] Step 2: Validating data...")
    validation = validate_financial_data(extracted)
    print(f"[AGENT] Validation: {json.dumps(validation, indent=2)}")
    
    # Step 3: Detect anomalies
    print("[AGENT] Step 3: Detecting anomalies...")
    anomalies = detect_anomalies(extracted, validation)
    print(f"[AGENT] Anomalies: {json.dumps(anomalies, indent=2)}")
    
    # Step 4: Audit log
    print("[AGENT] Step 4: Logging to audit trail...")
    doc_summary = document_text[:200]
    log_id = log_to_db(
        doc_summary,
        extracted,
        validation,
        validation.get("confidence", 0.5)
    )
    print(f"[AGENT] Logged with ID: {log_id}")
    
    # Compile results
    return {
        "log_id": log_id,
        "extracted_data": extracted,
        "validation_results": validation,
        "anomalies": anomalies,
        "status": "processed"
    }

# ============ RAG INTEGRATION ============
def process_document_with_rag(document_text: str, use_rag: bool = True) -> dict:
    """Process document with optional RAG enhancement."""
    from rag import enhance_extraction_with_rag, store_document_pattern, init_rag
    from datetime import datetime
    
    print("\n[AGENT] Starting document processing with RAG...")
    
    # Initialize RAG
    if use_rag:
        rag_ready = init_rag()
        if not rag_ready:
            print("[AGENT] RAG unavailable, proceeding without it")
            use_rag = False
    
    # Step 1: Extract
    print("[AGENT] Step 1: Extracting financial data...")
    extracted = extract_financial_data(document_text)
    print(f"[AGENT] Extracted: {json.dumps(extracted, indent=2)}")
    
    # Step 2: RAG Enhancement (optional)
    if use_rag:
        print("[AGENT] Step 2a: Enhancing with RAG...")
        extracted = enhance_extraction_with_rag(extracted, document_text)
    
    # Step 3: Validate
    print("[AGENT] Step 3: Validating data...")
    validation = validate_financial_data(extracted)
    print(f"[AGENT] Validation: {json.dumps(validation, indent=2)}")
    
    # Step 4: Detect anomalies
    print("[AGENT] Step 4: Detecting anomalies...")
    anomalies = detect_anomalies(extracted, validation)
    print(f"[AGENT] Anomalies: {json.dumps(anomalies, indent=2)}")
    
    # Step 5: Audit log
    print("[AGENT] Step 5: Logging to audit trail...")
    doc_summary = document_text[:200]
    log_id = log_to_db(
        doc_summary,
        extracted,
        validation,
        validation.get("confidence", 0.5)
    )
    print(f"[AGENT] Logged with ID: {log_id}")
    
    # Step 6: Store in RAG (optional)
    if use_rag:
        print("[AGENT] Step 6: Storing pattern in RAG...")
        store_document_pattern(document_text, extracted, log_id)
    
    # Compile results
    return {
        "log_id": log_id,
        "extracted_data": extracted,
        "validation_results": validation,
        "anomalies": anomalies,
        "status": "processed",
        "rag_enabled": use_rag
    }
