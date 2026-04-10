# Contract Intelligence App

A natural language contract Q&A system for oil & gas procurement teams, built as an agentic LLM application. Business operations and procurement staff can query a corpus of executed contracts in plain English — asking about termination terms, liability caps, risk flags, renewal provisions, and more — without needing to read through each contract manually.

## Live Demo

[contract-intelligence-app.vercel.app](https://contract-intelligence-app.vercel.app)

## Architecture

The system has three layers:

**Ingestion** — Contract `.txt` files are chunked, embedded using Amazon Titan Embeddings via AWS Bedrock, and stored in a pgvector table on RDS. Structured metadata (dates, parties, governing law, liability caps) is extracted using Llama 3 70B and stored in a separate metadata table for fast lookup.

**Agent** — An AWS Lambda function runs a Llama 3 70B agent (via Bedrock) that dynamically routes each user query through one or more tools before generating a response. The agent decides which tool to call based on the question type — it does not follow a fixed pipeline.

**Frontend** — A React web app deployed on Vercel. Users type natural language questions and receive cited, plain-English answers with tool call badges showing what the agent did.

```
User → React (Vercel)
     → API Gateway → Lambda (Agent + Tools)
                   → Bedrock (Llama 3 70B, Titan Embeddings)
                   → RDS PostgreSQL (pgvector + metadata + agent_logs)
```

## Agent Tools

The LLM selects among four tools per query:

| Tool | Purpose |
|------|---------|
| `search_contracts` | Semantic search across all contract chunks via pgvector |
| `lookup_structured_field` | Precise lookup of structured fields (dates, parties, governing law) |
| `compare_contracts` | Cross-contract comparison for a given clause type |
| `escalate_to_legal` | Flags questions requiring legal interpretation for human review |

## Stack

- **LLM**: Meta Llama 3 70B Instruct via AWS Bedrock
- **Embeddings**: Amazon Titan Embed Text v2 via AWS Bedrock
- **Vector store**: pgvector extension on AWS RDS (PostgreSQL 16)
- **Backend**: Python, AWS Lambda, AWS API Gateway
- **Frontend**: React 18, deployed on Vercel
- **Observability**: Custom logging table (`agent_logs`) on RDS

## Repository Structure

```
contract-intelligence-app/
├── README.md
├── .env.example                  ← environment variable template
├── requirements.txt              ← Python dependencies
├── backend/
│   ├── agent.py                  ← Lambda handler + agent loop + tools
│   ├── ingest.py                 ← contract ingestion pipeline
│   ├── test_retrieval.py         ← retrieval verification script
│   └── db_setup.sql              ← RDS schema
├── frontend/
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js
│       └── App.css
└── data/
    ├── contracts/                ← 25 synthetic Ergon contracts (.txt)
    └── eval/
        ├── ergon_eval_set.md     ← 50 ground truth Q&A pairs
        └── eval_results.json     ← scored eval run results
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. Never commit `.env` or `.env.local`.

```
DB_HOST=your-rds-endpoint.us-east-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=
AWS_REGION=us-east-1
REACT_APP_API_URL=https://uo8ng3v85j.execute-api.us-east-1.amazonaws.com/query
```

## Prerequisites

- AWS account with Bedrock model access enabled for:
  - `meta.llama3-70b-instruct-v1:0`
  - `amazon.titan-embed-text-v2:0`
- AWS RDS PostgreSQL 16 instance with pgvector extension
- AWS Lambda function with VPC access to RDS and VPC endpoints for Bedrock
- Python 3.12
- Node.js 18+

## Setup

### 1. Database

Connect to your RDS instance and run the schema:

```bash
psql -h your-rds-endpoint -U postgres -d postgres -f backend/db_setup.sql
```

### 2. Ingest contracts

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in your values in .env
python backend/ingest.py
```

This chunks and embeds all 25 contracts (~3-4 minutes due to Bedrock rate limits).

### 3. Verify retrieval

```bash
python backend/test_retrieval.py
```

All 5 tests should pass before proceeding.

### 4. Run agent locally

```bash
python backend/agent.py
```

Runs 5 test queries against your live RDS instance and Bedrock.

### 5. Deploy backend to Lambda

```bash
mkdir lambda_package
pip install psycopg2-binary boto3 -t lambda_package/ \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 3.12 \
  --only-binary=:all:
cp backend/agent.py lambda_package/
cd lambda_package && zip -r ../lambda_deployment.zip . && cd ..
```

Upload `lambda_deployment.zip` to AWS Lambda. Set all environment variables in Lambda configuration. Configure VPC with access to RDS and VPC endpoints for `bedrock` and `bedrock-runtime`.

Create an HTTP API in API Gateway with route `POST /query` pointing to your Lambda.

### 6. Run frontend locally

```bash
cd frontend
npm install
echo "REACT_APP_API_URL=https://uo8ng3v85j.execute-api.us-east-1.amazonaws.com/query" > .env.local
npm start
```

If you are replicating the AWS architecture in your own account, make sure to replace `uo8ng3v85j` with your API ID.

### 7. Deploy frontend to Vercel

Push the repo to GitHub. Go to vercel.com → New Project → import the repo → set environment variable `REACT_APP_API_URL` → Deploy.

## Evaluation

The system was evaluated against 22 representative questions across 8 categories drawn from a 50-question ground truth eval set. Results:

- Fully correct: 50% (11/22)
- Partially correct: 36% (8/22)  
- Wrong: 14% (3/22)

The system performs strongest on clause-level understanding (100% correct) and risk identification (67% correct). Cross-document multi-hop queries are the weakest category. See `data/eval/` for full results.

## Observability

Every agent request writes a record to the `agent_logs` RDS table capturing: `request_id`, `user_query`, `tool_called`, `tool_input`, `tool_output`, `final_response`, `latency_ms`, `escalated`, `created_at`.

Query logs directly:

```sql
SELECT tool_called, COUNT(*), ROUND(AVG(latency_ms))
FROM agent_logs
GROUP BY tool_called;
```

## Acknowledgements

Synthetic contract data generated using Claude (Anthropic). Provided context regarding the business segments in Ergon based on publicly available data. Project built for DS 5730 Final Project.
