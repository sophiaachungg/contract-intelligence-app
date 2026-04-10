"""
test_retrieval.py — Verify pgvector semantic search before building the agent.

Run this BEFORE writing any agent code. If this doesn't work, the agent won't work.

Usage:
    python test_retrieval.py
"""

import os
import json
import boto3
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "postgres"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
bedrock     = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def embed_text(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def semantic_search(conn, query: str, contract_id: str = None, top_k: int = 3):
    """Search contract_chunks by semantic similarity."""
    embedding = embed_text(query)
    vec_str   = "[" + ",".join(str(x) for x in embedding) + "]"

    if contract_id:
        sql = """
            SELECT contract_id, chunk_index, chunk_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM contract_chunks
            WHERE contract_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, contract_id, vec_str, top_k)
    else:
        sql = """
            SELECT contract_id, chunk_index, chunk_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM contract_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, vec_str, top_k)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def lookup_field(conn, contract_id: str, field: str):
    """Look up a structured metadata field for a contract."""
    valid_fields = [
        "contract_type", "branch", "counterparty", "effective_date",
        "expiration_date", "auto_renewal", "renewal_notice_days",
        "governing_law", "liability_cap", "termination_notice_days",
        "has_risk_flags", "raw_fields"
    ]
    if field not in valid_fields:
        return f"Unknown field '{field}'. Valid fields: {valid_fields}"

    sql = f"SELECT {field} FROM contract_metadata WHERE contract_id = %s;"
    with conn.cursor() as cur:
        cur.execute(sql, (contract_id,))
        row = cur.fetchone()
    return row[0] if row else None


def run_tests(conn):
    print("\n" + "="*60)
    print("TEST 1: Broad semantic search — liability cap question")
    print("="*60)
    results = semantic_search(conn, "what is the liability cap", top_k=3)
    for contract_id, chunk_idx, text, sim in results:
        print(f"\n  [{contract_id}] chunk {chunk_idx} (similarity: {sim:.3f})")
        print(f"  {text[:200]}...")

    print("\n" + "="*60)
    print("TEST 2: Single-contract search — spill liability in ISL-008")
    print("="*60)
    results = semantic_search(
        conn,
        "who is responsible for spill response costs",
        contract_id="ERG-ISL-2024-008",
        top_k=2
    )
    for contract_id, chunk_idx, text, sim in results:
        print(f"\n  [{contract_id}] chunk {chunk_idx} (similarity: {sim:.3f})")
        print(f"  {text[:300]}...")

    print("\n" + "="*60)
    print("TEST 3: Cross-contract search — force majeure")
    print("="*60)
    results = semantic_search(conn, "force majeure economic hardship termination", top_k=3)
    for contract_id, chunk_idx, text, sim in results:
        print(f"\n  [{contract_id}] chunk {chunk_idx} (similarity: {sim:.3f})")
        print(f"  {text[:200]}...")

    print("\n" + "="*60)
    print("TEST 4: Structured field lookup — expiration dates")
    print("="*60)
    for cid in ["ERG-ISL-2024-008", "ERG-ESS-2024-002", "ERG-CORP-2024-015"]:
        exp = lookup_field(conn, cid, "expiration_date")
        counterparty = lookup_field(conn, cid, "counterparty")
        print(f"  {cid}: expires {exp}, counterparty: {counterparty}")

    print("\n" + "="*60)
    print("TEST 5: Risk flag lookup")
    print("="*60)
    sql = """
        SELECT contract_id, counterparty, has_risk_flags
        FROM contract_metadata
        WHERE has_risk_flags = TRUE
        ORDER BY contract_id;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    print(f"  Contracts with risk flags: {len(rows)}")
    for cid, counterparty, _ in rows:
        print(f"  {cid} — {counterparty}")

    print("\n" + "="*60)
    print("All tests complete.")
    print("="*60)
    print("\nWhat to look for:")
    print("  TEST 1-3: Are the right contracts surfacing? Similarity > 0.5 is good.")
    print("  TEST 4:   Are dates and counterparties correct?")
    print("  TEST 5:   Should be all 25 contracts (all have RISK FLAG markers).")


def main():
    print("Connecting to RDS...")
    conn = connect_db()
    print("Connected.")
    run_tests(conn)
    conn.close()


def connect_db():
    return psycopg2.connect(**DB_CONFIG)


if __name__ == "__main__":
    main()
