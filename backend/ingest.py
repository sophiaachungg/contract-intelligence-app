"""
ingest.py — Ergon Contract Intelligence
Reads .txt contracts, chunks them, embeds with Titan via Bedrock,
extracts structured metadata with Llama 3, writes everything to RDS/pgvector.

Usage:
    python ingest.py

Requirements:
    pip install psycopg2-binary boto3 python-dotenv

Environment variables (put in .env):
    DB_HOST=your-rds-endpoint.rds.amazonaws.com
    DB_PORT=5432
    DB_NAME=postgres
    DB_USER=postgres
    DB_PASSWORD=your-password
    AWS_REGION=us-east-1
"""

import os
import re
import json
import time
import uuid
import boto3
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "postgres"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

AWS_REGION          = os.getenv("AWS_REGION", "us-east-1")
CONTRACTS_DIR       = "./contracts"
CHUNK_SIZE          = 150      # words per chunk
CHUNK_OVERLAP       = 30       # word overlap between chunks
EMBED_MODEL         = "amazon.titan-embed-text-v2:0"
LLAMA_MODEL         = "meta.llama3-70b-instruct-v1:0"
EMBED_DIMENSION     = 1024

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ── Helpers ───────────────────────────────────────────────────────────────────

def connect_db():
    return psycopg2.connect(**DB_CONFIG)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def embed_text(text: str) -> list[float]:
    """Call Titan Embeddings via Bedrock and return the embedding vector."""
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    body = json.loads(response["body"].read())
    return body["embedding"]


def extract_metadata_with_llama(contract_id: str, full_text: str) -> dict:
    """
    Call Llama 3 to extract structured fields from the contract text.
    Returns a dict of key fields. Falls back to regex for critical fields.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a contract analyst. Extract structured fields from the contract text below.
Return ONLY valid JSON with exactly these keys. Use null if a field is not found.
Do not include any explanation or text outside the JSON object.

Fields to extract:
- contract_type (string)
- branch (string: ESS, PCR, ISL, or CORP)
- counterparty (string: the other party's name)
- effective_date (string: YYYY-MM-DD format, or null)
- expiration_date (string: YYYY-MM-DD format, or null)
- auto_renewal (boolean)
- renewal_notice_days (integer or null)
- governing_law (string)
- liability_cap (string describing the cap, or null)
- termination_notice_days (integer or null)
- has_risk_flags (boolean: true if the text contains RISK FLAG markers)
<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTRACT TEXT:
{full_text[:3000]}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    try:
        response = bedrock.invoke_model(
            modelId=LLAMA_MODEL,
            body=json.dumps({
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.0,
            }),
            contentType="application/json",
            accept="application/json",
        )
        raw = json.loads(response["body"].read())
        text_out = raw.get("generation", "")

        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', text_out, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            print(f"  [WARN] Could not parse JSON from Llama response for {contract_id}")
            return {}

    except Exception as e:
        print(f"  [WARN] Llama metadata extraction failed for {contract_id}: {e}")
        return {}


def parse_header_fields(text: str) -> dict:
    """
    Regex fallback to extract header fields from our synthetic contract format.
    Used to fill gaps if Llama extraction fails.
    """
    def find(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    raw = {
        "contract_id":   find(r"CONTRACT ID:\s*(.+)"),
        "contract_type": find(r"TYPE:\s*(.+)"),
        "branch_raw":    find(r"BRANCH:\s*(.+)"),
        "effective_raw": find(r"EFFECTIVE DATE:\s*(.+)"),
        "expiration_raw":find(r"EXPIRATION DATE:\s*(.+)"),
        "auto_renewal":  find(r"AUTO-RENEWAL:\s*(.+)"),
    }

    # Normalize branch code
    branch_map = {"ESS": "ESS", "PCR": "PCR", "ISL": "ISL", "CORP": "CORP"}
    branch = None
    if raw["branch_raw"]:
        for code in branch_map:
            if code in raw["branch_raw"].upper():
                branch = code
                break

    # Normalize dates (Month DD, YYYY → YYYY-MM-DD)
    month_map = {
        "january":"01","february":"02","march":"03","april":"04",
        "may":"05","june":"06","july":"07","august":"08",
        "september":"09","october":"10","november":"11","december":"12"
    }
    def parse_date(s):
        if not s:
            return None
        m = re.match(r"(\w+)\s+(\d+),\s+(\d{4})", s)
        if m:
            mo = month_map.get(m.group(1).lower())
            if mo:
                return f"{m.group(3)}-{mo}-{m.group(2).zfill(2)}"
        return None

    # Auto-renewal notice days
    renewal_days = None
    if raw["auto_renewal"] and "yes" in raw["auto_renewal"].lower():
        m = re.search(r"(\d+)-day", raw["auto_renewal"], re.IGNORECASE)
        if m:
            renewal_days = int(m.group(1))

    return {
        "branch":               branch,
        "effective_date":       parse_date(raw["effective_raw"]),
        "expiration_date":      parse_date(raw["expiration_raw"]),
        "auto_renewal":         raw["auto_renewal"] and "yes" in raw["auto_renewal"].lower(),
        "renewal_notice_days":  renewal_days,
        "has_risk_flags":       "**RISK FLAG" in text,
    }


def merge_metadata(llama_meta: dict, header_meta: dict, contract_id: str) -> dict:
    """Merge Llama extraction with regex fallback. Regex wins for dates/flags."""
    merged = {
        "contract_id":              contract_id,
        "contract_type":            llama_meta.get("contract_type"),
        "branch":                   header_meta.get("branch") or llama_meta.get("branch"),
        "counterparty":             llama_meta.get("counterparty"),
        "effective_date":           header_meta.get("effective_date") or llama_meta.get("effective_date"),
        "expiration_date":          header_meta.get("expiration_date") or llama_meta.get("expiration_date"),
        "auto_renewal":             header_meta.get("auto_renewal"),
        "renewal_notice_days":      header_meta.get("renewal_notice_days") or llama_meta.get("renewal_notice_days"),
        "governing_law":            llama_meta.get("governing_law"),
        "liability_cap":            llama_meta.get("liability_cap"),
        "termination_notice_days":  llama_meta.get("termination_notice_days"),
        "has_risk_flags":           header_meta.get("has_risk_flags", False),
        "raw_fields":               json.dumps(llama_meta),
    }
    return merged


# ── Database writers ──────────────────────────────────────────────────────────

def upsert_metadata(conn, meta: dict):
    sql = """
        INSERT INTO contract_metadata (
            contract_id, contract_type, branch, counterparty,
            effective_date, expiration_date, auto_renewal, renewal_notice_days,
            governing_law, liability_cap, termination_notice_days,
            has_risk_flags, raw_fields
        ) VALUES (
            %(contract_id)s, %(contract_type)s, %(branch)s, %(counterparty)s,
            %(effective_date)s, %(expiration_date)s, %(auto_renewal)s, %(renewal_notice_days)s,
            %(governing_law)s, %(liability_cap)s, %(termination_notice_days)s,
            %(has_risk_flags)s, %(raw_fields)s
        )
        ON CONFLICT (contract_id) DO UPDATE SET
            contract_type           = EXCLUDED.contract_type,
            branch                  = EXCLUDED.branch,
            counterparty            = EXCLUDED.counterparty,
            effective_date          = EXCLUDED.effective_date,
            expiration_date         = EXCLUDED.expiration_date,
            auto_renewal            = EXCLUDED.auto_renewal,
            renewal_notice_days     = EXCLUDED.renewal_notice_days,
            governing_law           = EXCLUDED.governing_law,
            liability_cap           = EXCLUDED.liability_cap,
            termination_notice_days = EXCLUDED.termination_notice_days,
            has_risk_flags          = EXCLUDED.has_risk_flags,
            raw_fields              = EXCLUDED.raw_fields;
    """
    with conn.cursor() as cur:
        cur.execute(sql, meta)
    conn.commit()


def insert_chunks(conn, contract_id: str, branch: str, chunks: list[str], embeddings: list[list[float]]):
    # Delete existing chunks for this contract (safe re-run)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM contract_chunks WHERE contract_id = %s", (contract_id,))

    rows = [
        (contract_id, branch, i, chunk, embedding)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    sql = """
        INSERT INTO contract_chunks (contract_id, branch, chunk_index, chunk_text, embedding)
        VALUES %s
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, template="(%s, %s, %s, %s, %s::vector)")
    conn.commit()


# ── Main ingestion loop ───────────────────────────────────────────────────────

def ingest_contract(conn, filepath: str):
    filename = os.path.basename(filepath)
    print(f"\n→ Ingesting {filename}")

    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Extract contract ID from header
    id_match = re.search(r"CONTRACT ID:\s*(\S+)", full_text)
    contract_id = id_match.group(1) if id_match else filename.replace(".txt", "")
    print(f"  Contract ID: {contract_id}")

    # Step 1: Extract metadata
    print("  Extracting metadata with Llama 3...")
    llama_meta  = extract_metadata_with_llama(contract_id, full_text)
    header_meta = parse_header_fields(full_text)
    meta        = merge_metadata(llama_meta, header_meta, contract_id)
    upsert_metadata(conn, meta)
    print(f"  Metadata written (counterparty: {meta.get('counterparty')}, expires: {meta.get('expiration_date')})")

    # Step 2: Chunk text
    chunks = chunk_text(full_text)
    print(f"  {len(chunks)} chunks created")

    # Step 3: Embed each chunk
    print("  Embedding chunks with Titan...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        embeddings.append(embedding)
        # Avoid Bedrock rate limiting
        time.sleep(0.1)

    # Step 4: Write chunks + embeddings to RDS
    branch = meta.get("branch") or "UNKNOWN"
    insert_chunks(conn, contract_id, branch, chunks, embeddings)
    print(f"  {len(chunks)} chunks + embeddings written to RDS")


def main():
    print("Connecting to RDS...")
    conn = connect_db()
    print("Connected.\n")

    contract_files = sorted([
        os.path.join(CONTRACTS_DIR, f)
        for f in os.listdir(CONTRACTS_DIR)
        if f.endswith(".txt")
    ])

    print(f"Found {len(contract_files)} contracts to ingest.")

    failed = []
    for filepath in contract_files:
        try:
            ingest_contract(conn, filepath)
        except Exception as e:
            print(f"  [ERROR] Failed to ingest {filepath}: {e}")
            failed.append(filepath)

    conn.close()

    print("\n" + "="*50)
    print(f"Ingestion complete. {len(contract_files) - len(failed)}/{len(contract_files)} succeeded.")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
