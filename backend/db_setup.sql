CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE contract_chunks (
    id          SERIAL PRIMARY KEY,
    contract_id VARCHAR(50) NOT NULL,
    branch      VARCHAR(10),
    chunk_index INTEGER,
    chunk_text  TEXT,
    embedding   vector(1024),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON contract_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

CREATE TABLE contract_metadata (
    id                      SERIAL PRIMARY KEY,
    contract_id             VARCHAR(50) UNIQUE NOT NULL,
    contract_type           TEXT,
    branch                  VARCHAR(10),
    counterparty            TEXT,
    effective_date          DATE,
    expiration_date         DATE,
    auto_renewal            BOOLEAN,
    renewal_notice_days     INTEGER,
    governing_law           TEXT,
    liability_cap           TEXT,
    termination_notice_days INTEGER,
    has_risk_flags          BOOLEAN,
    raw_fields              JSONB,
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE TABLE agent_logs (
    id             SERIAL PRIMARY KEY,
    request_id     VARCHAR(50),
    user_query     TEXT,
    tool_called    VARCHAR(50),
    tool_input     JSONB,
    tool_output    TEXT,
    final_response TEXT,
    latency_ms     INTEGER,
    escalated      BOOLEAN DEFAULT FALSE,
    created_at     TIMESTAMP DEFAULT NOW()
);
