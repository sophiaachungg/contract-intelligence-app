import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'https://your-api-id.execute-api.us-east-1.amazonaws.com/query';

const SUGGESTED_QUERIES = [
  "Which contracts expire within the next 12 months?",
  "Compare liability caps across ISL contracts",
  "What are the risk flags in our crude gathering agreement with Tombigbee?",
  "Which contracts have auto-renewal provisions?",
  "Can our German customer source elsewhere if we can't deliver?",
];

function ToolCallBadge({ tool }) {
  const icons = {
    search_contracts: '⌕',
    lookup_structured_field: '≡',
    compare_contracts: '⇄',
    escalate_to_legal: '⚑',
  };
  return (
    <span className="tool-badge">
      <span className="tool-icon">{icons[tool] || '○'}</span>
      {tool.replace(/_/g, ' ')}
    </span>
  );
}

function Message({ msg }) {
  return (
    <div className={`message message--${msg.role}`}>
      {msg.role === 'assistant' && (
        <div className="message-meta">
          <span className="message-label">ERGON CI</span>
          {msg.toolCalls && msg.toolCalls.length > 0 && (
            <div className="tool-calls">
              {msg.toolCalls.map((t, i) => (
                <ToolCallBadge key={i} tool={t.tool} />
              ))}
            </div>
          )}
          {msg.escalated && (
            <span className="escalated-badge">⚑ Escalated to Legal</span>
          )}
          {msg.latency && (
            <span className="latency">{(msg.latency / 1000).toFixed(1)}s</span>
          )}
        </div>
      )}
      {msg.role === 'user' && (
        <div className="message-meta">
          <span className="message-label">YOU</span>
        </div>
      )}
      <div className="message-content">
        {msg.role === 'error' ? (
          <span className="error-text">{msg.content}</span>
        ) : (
          msg.content
        )}
      </div>
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="message message--assistant">
      <div className="message-meta">
        <span className="message-label">ERGON CI</span>
      </div>
      <div className="message-content thinking">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  );
}

export default function App() {
  const [dark, setDark] = useState(true);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }, [dark]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  };

  const sendQuery = async (query) => {
    if (!query.trim() || loading) return;
    const userMsg = { role: 'user', content: query };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        toolCalls: data.tool_calls || [],
        escalated: data.escalated,
        latency: data.latency_ms,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'error',
        content: `Request failed: ${err.message}`,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery(input);
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <span className="wordmark">ERGON</span>
          <span className="wordmark-sub">Contract Intelligence</span>
        </div>
        <button
          className="theme-toggle"
          onClick={() => setDark(d => !d)}
          aria-label="Toggle theme"
        >
          {dark ? '○' : '●'}
        </button>
      </header>

      {/* Main */}
      <main className="main">
        {isEmpty && (
          <div className="empty-state">
            <div className="empty-eyebrow">25 contracts indexed</div>
            <h1 className="empty-heading">
              Ask anything about<br />your contracts.
            </h1>
            <div className="suggestions">
              {SUGGESTED_QUERIES.map((q, i) => (
                <button
                  key={i}
                  className="suggestion"
                  onClick={() => sendQuery(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {!isEmpty && (
          <div className="messages">
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} />
            ))}
            {loading && <ThinkingIndicator />}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* Input */}
      <footer className="footer">
        <div className="input-row">
          <textarea
            ref={textareaRef}
            className="input"
            value={input}
            onChange={e => { setInput(e.target.value); autoResize(); }}
            onKeyDown={handleKeyDown}
            placeholder="Ask about termination terms, liability caps, risk flags…"
            rows={1}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={() => sendQuery(input)}
            disabled={loading || !input.trim()}
            aria-label="Send"
          >
            ↑
          </button>
        </div>
        <div className="footer-hint">
          Enter to send · Shift+Enter for new line · Responses cite contract IDs
        </div>
      </footer>
    </div>
  );
}
