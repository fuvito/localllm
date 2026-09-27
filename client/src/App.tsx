import { useState, useRef, useEffect } from 'react'
import './App.css'

type Message = {
  role: 'user' | 'assistant'
  content: string
}

const API_URL = 'http://localhost:8000'

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Could not reach the server. Is it running?' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className={`chat-container ${fullscreen ? 'fullscreen' : ''}`}>
      <header className="chat-header">
        <span className="logo">🍕</span>
        <h1>Luigi's Pizza</h1>
        <button
          className="fullscreen-btn"
          onClick={() => setFullscreen(f => !f)}
          title={fullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
        >
          {fullscreen ? '⤓' : '⤢'}
        </button>
      </header>

      <div className="messages">
        <div className="message assistant">
          <span>Welcome to Luigi's Pizza! How can I help you today?</span>
        </div>
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <span>{m.content}</span>
          </div>
        ))}
        {loading && (
          <div className="message assistant loading">
            <span className="dots">
              <span>.</span><span>.</span><span>.</span>
            </span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="input-row">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about our menu, hours, or dietary info…"
          disabled={loading}
          autoFocus
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}
