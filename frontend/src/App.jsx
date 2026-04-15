import { useState, useRef, useEffect } from "react";
import "./App.css";

const SESSION_ID = crypto.randomUUID();
const API = "http://localhost:8001";

const SUGGESTIONS = [
  { en: "When is registration?",     ar: "متى التسجيل؟" },
  { en: "When are finals?",          ar: "متى الفاينل؟" },
  { en: "When do classes start?",    ar: "متى تبدأ الدراسة؟" },
  { en: "What are the holidays?",    ar: "ما هي الإجازات؟" },
  { en: "When is Eid Al-Fitr?",      ar: "متى عيد الفطر؟" },
  { en: "When is the drop deadline?",ar: "متى آخر موعد للحذف؟" },
];

function Message({ msg }) {
  const isBot = msg.role === "bot";
  return (
    <div className={`message ${msg.role}`}>
      <div className="bubble">
        <p>{msg.text}</p>
        {isBot && msg.source && <span className="source">{msg.source}</span>}
        {msg.warning && <span className="warning">{msg.warning}</span>}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message bot">
      <div className="bubble typing">
        <span /><span /><span />
      </div>
    </div>
  );
}

function Suggestions({ onSelect }) {
  return (
    <div className="suggestions">
      <p className="suggestions-label">Quick questions / أسئلة سريعة</p>
      <div className="suggestions-grid">
        {SUGGESTIONS.map((s, i) => (
          <button key={i} className="suggestion-chip" onClick={() => onSelect(s.en)}>
            <span className="chip-en">{s.en}</span>
            <span className="chip-ar">{s.ar}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "أهلاً! أنا مساعد تقويم جامعة البحرين.\nHello! I'm the UOB Calendar AI. Ask me anything about the 2025/2026 academic calendar.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [rateError, setRateError] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function addBotMessage(text, source = null, warning = null) {
    setMessages((prev) => [...prev, { role: "bot", text, source, warning }]);
  }

  async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    setShowSuggestions(false);
    setRateError(null);
    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
    setInput("");
    setLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, session_id: SESSION_ID }),
      });

      const data = await res.json();

      if (res.status === 429) {
        setRateError(data.detail);
      } else if (!res.ok) {
        addBotMessage(data.detail || "Something went wrong. Please try again.");
      } else {
        addBotMessage(data.response, data.source, data.warning);
      }
    } catch {
      addBotMessage("Could not reach the server. Make sure the API is running.");
    }

    setLoading(false);
  }

  async function clearHistory() {
    try {
      await fetch(`${API}/clear`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID }),
      });
    } catch {
      // ignore
    }
    setRateError(null);
    setShowSuggestions(true);
    setMessages([
      {
        role: "bot",
        text: "Conversation cleared. / تم مسح المحادثة.",
      },
    ]);
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  function handleInput(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">🎓</div>
            <div className="logo-text">
              <h1>UOB Calendar AI</h1>
              <p>جامعة البحرين — Academic Calendar 2025/2026</p>
            </div>
          </div>
          <button className="new-chat-btn" onClick={clearHistory}>
            New Chat
          </button>
        </div>
      </header>

      <main className="chat-window">
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {showSuggestions && !loading && (
          <Suggestions onSelect={sendMessage} />
        )}
        {loading && <TypingIndicator />}
        {rateError && <div className="rate-error">{rateError}</div>}
        <div ref={bottomRef} />
      </main>

      <footer className="input-bar">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKey}
          placeholder="Ask about exams, holidays, registration… / اسأل عن المواعيد والإجازات..."
          rows={1}
          disabled={loading}
        />
        <button
          className="send-btn"
          onClick={() => sendMessage(input)}
          disabled={loading || !input.trim()}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </footer>
    </div>
  );
}
