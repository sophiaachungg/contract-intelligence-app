"""
agent.py — Ergon Contract Intelligence Agent
Lambda handler + agent loop using Llama 3 70B on Bedrock with 4 tools.

Tools:
  search_contracts(query, contract_id)   — semantic search via pgvector
  lookup_structured_field(contract_id, field) — structured metadata lookup
  compare_contracts(contract_ids, clause_type) — cross-contract comparison
  escalate_to_legal(reason)              — flags question for human review

Observability:
  Every request writes a record to agent_logs table in RDS.

Environment variables:
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  AWS_REGION
"""


import os
import re
import json
import time
import uuid
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
LLAMA_MODEL = "meta.llama3-70b-instruct-v1:0"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
TOP_K       = 4      # chunks to retrieve per search
MAX_TURNS   = 6      # max agent loop iterations before forcing a response

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "postgres"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

# Reuse DB connection across warm Lambda invocations
_conn = None

def get_db():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(**DB_CONFIG)
    return _conn


# ── Tool definitions (JSON schema Llama reads) ────────────────────────────────

TOOLS = [
    {
        "name": "search_contracts",
        "description": (
            "Perform semantic search across contract text. Use this when the user asks "
            "an open-ended question about contract content, clauses, obligations, risks, "
            "or terms — especially when you don't know which contract contains the answer. "
            "Optionally filter to a specific contract_id if the user mentions one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — rephrase the user question as a short declarative statement about contract content."
                },
                "contract_id": {
                    "type": "string",
                    "description": "Optional. Filter search to a specific contract ID (e.g. ERG-ISL-2024-008). Omit to search all contracts."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "lookup_structured_field",
        "description": (
            "Look up a specific structured field for a known contract. Use this for precise "
            "factual questions about dates, parties, renewal terms, governing law, or liability "
            "caps — where you already know the contract ID. Much faster and more reliable than "
            "semantic search for these fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contract_id": {
                    "type": "string",
                    "description": "The contract ID to look up (e.g. ERG-ISL-2024-008)."
                },
                "field": {
                    "type": "string",
                    "description": (
                        "The field to retrieve. Valid values: contract_type, branch, counterparty, "
                        "effective_date, expiration_date, auto_renewal, renewal_notice_days, "
                        "governing_law, liability_cap, termination_notice_days, has_risk_flags."
                    )
                }
            },
            "required": ["contract_id", "field"]
        }
    },
    {
        "name": "compare_contracts",
        "description": (
            "Compare a specific clause type or topic across multiple contracts. Use this when "
            "the user asks 'which contracts have X' or 'compare the termination terms across "
            "our ISL contracts' or similar cross-contract questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contract_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of contract IDs to compare. If empty, searches across all contracts."
                },
                "clause_type": {
                    "type": "string",
                    "description": "The clause or topic to compare across contracts (e.g. 'termination notice', 'force majeure', 'liability cap')."
                }
            },
            "required": ["clause_type"]
        }
    },
    {
        "name": "escalate_to_legal",
        "description": (
            "Flag this question for human legal review. Use this when: the question requires "
            "legal interpretation rather than factual retrieval, the answer has significant "
            "legal or financial consequences, the contract language is ambiguous, or the user "
            "is asking for a legal opinion. Do NOT use for straightforward factual questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this question requires legal review."
                }
            },
            "required": ["reason"]
        }
    }
]

TOOLS_DESCRIPTION = "\n".join([
    f"- {t['name']}: {t['description'].split('.')[0]}."
    for t in TOOLS
])


# ── Tool implementations ──────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def tool_search_contracts(query: str, contract_id: str = None) -> str:
    conn = get_db()
    embedding = embed_text(query)
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"

    if contract_id:
        sql = """
            SELECT contract_id, chunk_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM contract_chunks
            WHERE contract_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, contract_id, vec_str, TOP_K)
    else:
        sql = """
            SELECT contract_id, chunk_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM contract_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, vec_str, TOP_K)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return "No relevant contract passages found."

    results = []
    for contract_id_res, chunk_text, similarity in rows:
        results.append(
            f"[{contract_id_res}] (relevance: {similarity:.2f})\n{chunk_text}"
        )
    return "\n\n---\n\n".join(results)


def tool_lookup_structured_field(contract_id: str, field: str) -> str:
    valid_fields = [
        "contract_type", "branch", "counterparty", "effective_date",
        "expiration_date", "auto_renewal", "renewal_notice_days",
        "governing_law", "liability_cap", "termination_notice_days",
        "has_risk_flags"
    ]
    if field not in valid_fields:
        return f"Invalid field '{field}'. Valid fields: {', '.join(valid_fields)}"

    conn = get_db()
    sql = f"SELECT {field} FROM contract_metadata WHERE contract_id = %s;"
    with conn.cursor() as cur:
        cur.execute(sql, (contract_id,))
        row = cur.fetchone()

    if not row:
        return f"No contract found with ID '{contract_id}'."
    value = row[0]
    if value is None:
        return f"Field '{field}' is not set for contract {contract_id}."
    return str(value)


def tool_compare_contracts(clause_type: str, contract_ids: list = None) -> str:
    conn = get_db()
    embedding = embed_text(clause_type)
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"

    # Llama sometimes passes contract_ids as a string instead of a list
    if isinstance(contract_ids, str):
        contract_ids = None

    if contract_ids:
        placeholders = ",".join(["%s"] * len(contract_ids))
        sql = f"""
            SELECT DISTINCT ON (contract_id)
                contract_id, chunk_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM contract_chunks
            WHERE contract_id IN ({placeholders})
            ORDER BY contract_id, embedding <=> %s::vector
            LIMIT 20;
        """
        params = [vec_str] + contract_ids + [vec_str]
    else:
        sql = """
            SELECT DISTINCT ON (contract_id)
                contract_id, chunk_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM contract_chunks
            ORDER BY contract_id, embedding <=> %s::vector
            LIMIT 25;
        """
        params = [vec_str, vec_str]

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return "No results found for comparison."

    # Sort by similarity descending
    rows = sorted(rows, key=lambda r: r[2], reverse=True)

    results = []
    for contract_id_res, chunk_text, similarity in rows:
        results.append(
            f"[{contract_id_res}] (relevance: {similarity:.2f})\n{chunk_text[:400]}"
        )
    return "\n\n---\n\n".join(results)


def tool_escalate_to_legal(reason: str) -> str:
    # In production this would write to a queue or send a notification
    # For now it returns a structured escalation message
    return (
        f"ESCALATED TO LEGAL REVIEW\n"
        f"Reason: {reason}\n"
        f"This question has been flagged and requires review by Ergon's legal team "
        f"before a definitive answer can be provided."
    )


def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to its implementation."""
    if tool_name == "search_contracts":
        return tool_search_contracts(
            query=tool_args["query"],
            contract_id=tool_args.get("contract_id")
        )
    elif tool_name == "lookup_structured_field":
        return tool_lookup_structured_field(
            contract_id=tool_args["contract_id"],
            field=tool_args["field"]
        )
    elif tool_name == "compare_contracts":
        return tool_compare_contracts(
            clause_type=tool_args["clause_type"],
            contract_ids=tool_args.get("contract_ids", [])
        )
    elif tool_name == "escalate_to_legal":
        return tool_escalate_to_legal(reason=tool_args["reason"])
    else:
        return f"Unknown tool: {tool_name}"


# ── Llama interface ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a contract intelligence assistant for Ergon, an oil, gas, asphalt, and logistics company.
You help procurement and operations staff understand executed contracts quickly and accurately.

You have access to these tools:
{TOOLS_DESCRIPTION}

Rules:
1. Always use a tool before answering — do not answer from memory alone.
2. For specific factual fields (dates, parties, governing law), prefer lookup_structured_field over search_contracts.
3. For open-ended questions about contract content, use search_contracts.
4. For questions spanning multiple contracts, use compare_contracts.
5. If a question requires legal interpretation or judgment, use escalate_to_legal.
6. After receiving tool results, synthesize a clear, concise answer for a procurement professional.
7. Always cite which contract(s) your answer comes from.
8. Never make up contract details. If you cannot find the answer, say so.
9. Do not provide legal advice. You surface contract facts; legal judgment belongs to Ergon's legal team.

To call a tool, respond with EXACTLY this format and nothing else:
TOOL_CALL: {{"tool": "tool_name", "args": {{"arg1": "value1"}}}}

After receiving tool results, provide your final answer in plain English.

CRITICAL RULES FOR TOOL USE:
- For lookup_structured_field, the ONLY valid fields are: contract_type, branch, counterparty, effective_date, expiration_date, auto_renewal, renewal_notice_days, governing_law, liability_cap, termination_notice_days, has_risk_flags. Use termination_notice_days, NOT termination_notice_period.
- For compare_contracts, the ONLY valid args are clause_type and contract_ids. No other args.
- For compare_contracts, contract_ids must be a JSON array of specific contract ID strings like ["ERG-ISL-2024-008", "ERG-ISL-2024-009"]. Never pass a descriptive string like "ISL contracts". If you want all ISL contracts, omit contract_ids entirely and filter by clause_type instead.
- After receiving a tool result, you MUST use the actual value from the result in your answer. Never write placeholder text like [insert value].
- If a tool returns an error, try a different field name or use search_contracts instead.

"""


def call_llama(messages: list) -> str:
    """Format conversation history and call Llama 3 on Bedrock."""
    # Build the prompt in Llama 3 chat format
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    response = bedrock.invoke_model(
        modelId=LLAMA_MODEL,
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": 1024,
            "temperature": 0.1,
        }),
        contentType="application/json",
        accept="application/json",
    )
    raw = json.loads(response["body"].read())
    return raw.get("generation", "").strip()


def parse_tool_call(text: str):
    """
    Parse a TOOL_CALL from Llama's response.
    Returns (tool_name, tool_args) or (None, None) if no tool call found.
    """
    match = re.search(r"TOOL_CALL:\s*(\{.*\})", text, re.DOTALL)
    if not match:
        return None, None
    try:
        parsed = json.loads(match.group(1))
        return parsed.get("tool"), parsed.get("args", {})
    except json.JSONDecodeError:
        return None, None


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_query: str) -> dict:
    """
    Main agent loop. Returns dict with final_response, tool_calls, escalated.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    messages = [{"role": "user", "content": user_query}]
    tool_call_log = []
    escalated = False
    final_response = None

    for turn in range(MAX_TURNS):
        llm_output = call_llama(messages)

        tool_name, tool_args = parse_tool_call(llm_output)

        if tool_name:
            # Execute tool
            print(f"[Turn {turn+1}] Tool call: {tool_name}({tool_args})")
            tool_result = dispatch_tool(tool_name, tool_args)

            tool_call_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result_preview": tool_result[:200]
            })

            if tool_name == "escalate_to_legal":
                escalated = True

            # Add tool call + result to conversation
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{tool_result}\n\nNow provide your final answer to the user."
            })

        else:
            # No tool call — treat as final response
            final_response = llm_output
            break

    if final_response is None:
        final_response = "I was unable to produce a complete answer. Please try rephrasing your question."

    latency_ms = int((time.time() - start_time) * 1000)

    # Write observability log
    try:
        log_request(
            request_id=request_id,
            user_query=user_query,
            tool_calls=tool_call_log,
            final_response=final_response,
            latency_ms=latency_ms,
            escalated=escalated,
        )
    except Exception as e:
        print(f"[WARN] Failed to write agent log: {e}")

    return {
        "request_id":     request_id,
        "final_response": final_response,
        "tool_calls":     tool_call_log,
        "escalated":      escalated,
        "latency_ms":     latency_ms,
    }


# ── Observability ─────────────────────────────────────────────────────────────

def log_request(request_id, user_query, tool_calls, final_response, latency_ms, escalated):
    conn = get_db()
    sql = """
        INSERT INTO agent_logs
            (request_id, user_query, tool_called, tool_input, tool_output,
             final_response, latency_ms, escalated)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    # Flatten tool calls into a single log row for simplicity
    tool_names   = ", ".join([t["tool"] for t in tool_calls]) if tool_calls else None
    tool_inputs  = json.dumps([t["args"] for t in tool_calls]) if tool_calls else None
    tool_outputs = json.dumps([t["result_preview"] for t in tool_calls]) if tool_calls else None

    with conn.cursor() as cur:
        cur.execute(sql, (
            request_id,
            user_query,
            tool_names,
            tool_inputs,
            tool_outputs,
            final_response[:2000],
            latency_ms,
            escalated,
        ))
    conn.commit()


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    Expects: { "query": "user question here" }
    Returns: { "response": "...", "tool_calls": [...], "escalated": bool, "latency_ms": int }
    """
    try:
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
        user_query = body.get("query", "").strip()

        if not user_query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'query' in request body"})
            }

        result = run_agent(user_query)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({
                "response":   result["final_response"],
                "tool_calls": result["tool_calls"],
                "escalated":  result["escalated"],
                "latency_ms": result["latency_ms"],
                "request_id": result["request_id"],
            })
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# ── Local test harness ────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What is the termination notice period in ERG-ISL-2024-008?",
        "Which contracts have force majeure clauses that include economic hardship?",
        "Compare the liability caps across our ISL contracts.",
        "Can our German customer source from someone else if we can't deliver?",
        "Should we be worried about anything in our crude gathering agreement with Tombigbee?",
    ]

    for query in test_queries:
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        result = run_agent(query)
        print(f"\nRESPONSE:\n{result['final_response']}")
        print(f"\nTOOLS USED: {[t['tool'] for t in result['tool_calls']]}")
        print(f"ESCALATED: {result['escalated']}")
        print(f"LATENCY: {result['latency_ms']}ms")
