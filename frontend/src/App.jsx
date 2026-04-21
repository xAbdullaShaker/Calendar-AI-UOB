import { useState, useRef, useEffect } from "react";
import "./App.css";
import botAvatar from "./assets/bot-avatar.png";

const SESSION_ID = crypto.randomUUID();
const API = import.meta.env.VITE_API_URL || "http://localhost:8001";

const SUGGESTIONS = [
  { en: "When does the semester begin?",          ar: "متى تبدأ الدراسة؟",                    icon: "📅" },
  { en: "When are the final examinations?",       ar: "متى الامتحانات النهائية؟",              icon: "📝" },
  { en: "When is the registration period?",       ar: "متى فترة التسجيل؟",                    icon: "🗂️" },
  { en: "What are the official public holidays?", ar: "ما هي الإجازات الرسمية؟",              icon: "🏖️" },
  { en: "When is the course withdrawal deadline?",ar: "متى آخر موعد الانسحاب من المقررات؟",  icon: "⏳" },
  { en: "When are examination results announced?",ar: "متى يُعلن عن نتائج الامتحانات؟",       icon: "🎓" },
];

function Message({ msg }) {
  const isBot = msg.role === "bot";
  return (
    <div className={`message ${msg.role}`}>
      {isBot && <img src={botAvatar} className="bot-avatar" alt="AI" />}
      <div className="bubble">
        <p>{msg.text}</p>
        {isBot && msg.source && (
          <span className={`source ${msg.source.startsWith("FAQ") ? "source-faq" : "source-rag"}`}>
            {msg.source}
          </span>
        )}
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

function Suggestions({ onSelect, lang, onLangToggle }) {
  return (
    <div className="suggestions">
      <div className="suggestions-header">
        <p className="suggestions-label">Quick questions / أسئلة سريعة</p>
        <div className="lang-toggle">
          <button
            className={`lang-btn ${lang === "en" ? "active" : ""}`}
            onClick={() => onLangToggle("en")}
          >EN</button>
          <button
            className={`lang-btn ${lang === "ar" ? "active" : ""}`}
            onClick={() => onLangToggle("ar")}
          >ع</button>
        </div>
      </div>
      <div className="suggestions-grid">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            className={`suggestion-chip ${lang === "ar" ? "rtl" : ""}`}
            onClick={() => onSelect(lang === "ar" ? s.ar : s.en)}
          >
            <span className="chip-icon">{s.icon}</span>
            {lang === "ar" ? s.ar : s.en}
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
  const [lang, setLang] = useState("en");
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

    setRateError(null);
    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
    setInput("");
    setLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    // Add an empty bot message we'll fill in token by token
    setMessages((prev) => [...prev, { role: "bot", text: "", source: null, warning: null }]);

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, session_id: SESSION_ID }),
      });

      if (res.status === 429) {
        const data = await res.json().catch(() => ({}));
        setMessages((prev) => prev.slice(0, -1));
        setRateError(data.detail || "Too many requests. Please wait a moment.");
        setLoading(false);
        return;
      }

      if (res.status === 400) {
        const data = await res.json().catch(() => ({}));
        setMessages((prev) => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = { role: "bot", text: data.detail || "Please enter a valid question." };
          return msgs;
        });
        setLoading(false);
        return;
      }

      if (!res.ok) {
        setMessages((prev) => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = { role: "bot", text: "Something went wrong. Please try again. / حدث خطأ، يرجى المحاولة مجدداً." };
          return msgs;
        });
        setLoading(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete last line in buffer

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "token") {
              setMessages((prev) => {
                const msgs = [...prev];
                const last = msgs[msgs.length - 1];
                msgs[msgs.length - 1] = { ...last, text: last.text + data.text };
                return msgs;
              });
            } else if (data.type === "done") {
              setMessages((prev) => {
                const msgs = [...prev];
                const last = msgs[msgs.length - 1];
                msgs[msgs.length - 1] = { ...last, source: data.source, warning: data.warning };
                return msgs;
              });
            }
          } catch {
            // skip malformed SSE event
          }
        }
      }
    } catch {
      setMessages((prev) => {
        const msgs = [...prev];
        msgs[msgs.length - 1] = { role: "bot", text: "Could not reach the server. Make sure the API is running." };
        return msgs;
      });
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
            <img
              className="logo-crest"
              src="https://www.uob.edu.bh/wp-content/uploads/site-prod/uploads/UOB-new-logo.png"
              alt="University of Bahrain"
            />
            <div className="logo-divider" />
            <div className="logo-text">
              <h1>University of Bahrain</h1>
              <span className="logo-sub">Academic Calendar AI — 2025/2026</span>
              <span className="logo-ar">مساعد التقويم الأكاديمي</span>
            </div>
          </div>
          <div className="header-right">
            <span className="ai-badge">GPT-4.1 mini</span>
            <button className="new-chat-btn" onClick={clearHistory}>New Chat</button>
          </div>
        </div>
      </header>

      <main className="chat-window">
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        <Suggestions onSelect={sendMessage} lang={lang} onLangToggle={setLang} />
        {loading && <TypingIndicator />}
        {rateError && <div className="rate-error">{rateError}</div>}
        <div ref={bottomRef} />
      </main>

      <footer className="input-bar">
        <div className="input-bar-inner">
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
        </div>
      </footer>
    </div>
  );
}
