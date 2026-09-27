import { useState, useRef, useEffect } from 'react'
import { useParams, useLocation, useNavigate } from 'react-router-dom'
import { API_URL, type Restaurant } from '../api'

type Message = {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

const fmtTime = (d: Date) =>
  d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true })

export default function ChatPage() {
  const { restaurantId } = useParams<{ restaurantId: string }>()
  const location = useLocation()
  const navigate = useNavigate()

  const [restaurant, setRestaurant] = useState<Restaurant | null>(
    location.state as Restaurant | null
  )
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const welcomeTime = useRef(new Date())

  // If navigated directly (no router state), fetch restaurant info
  useEffect(() => {
    if (!restaurant) {
      fetch(`${API_URL}/restaurants`)
        .then(r => r.json())
        .then((list: Restaurant[]) => {
          const found = list.find(r => r.id === restaurantId)
          if (found) setRestaurant(found)
          else navigate('/')
        })
        .catch(() => navigate('/'))
    }
  }, [])

  // Create a session when the page loads
  useEffect(() => {
    if (!restaurantId) return
    fetch(`${API_URL}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ restaurant_id: restaurantId }),
    })
      .then(r => r.json())
      .then(data => setSessionId(data.session_id))
      .catch(() => navigate('/'))
  }, [restaurantId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Focus input after response arrives (after input is re-enabled)
  useEffect(() => {
    if (!loading) inputRef.current?.focus()
  }, [loading])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading || !sessionId) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: text, timestamp: new Date() }])
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply, timestamp: new Date() }])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Could not reach the server. Is it running?', timestamp: new Date() },
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

  const downloadTranscript = () => {
    const name = restaurant?.name ?? 'Assistant'
    const date = new Date().toLocaleString()
    const lines = [
      `Chat with ${name}`,
      `Date: ${date}`,
      '',
      `[${fmtTime(welcomeTime.current)}] ${name}: Welcome to ${name}! How can I help you today?`,
      ...messages.map(m =>
        m.role === 'assistant'
          ? `[${fmtTime(m.timestamp)}] ${name}: ${m.content}`
          : `[${fmtTime(m.timestamp)}] You: ${m.content}`
      ),
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `chat-${name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className={`chat-container ${fullscreen ? 'fullscreen' : ''}`}>
      <header className="chat-header">
        <button className="back-btn" onClick={() => navigate('/')} title="Back to home">
          ‹
        </button>
        <span className="logo">{restaurant?.icon ?? '🍽️'}</span>
        <h1>{restaurant?.name ?? '…'}</h1>
        <button
          className="fullscreen-btn"
          onClick={downloadTranscript}
          disabled={messages.length === 0}
          title="Download transcript"
        >
          ⬇
        </button>
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
          <div className="message-bubble">
            <span>Welcome to {restaurant?.name ?? 'our restaurant'}! How can I help you today?</span>
            <time className="message-time">{fmtTime(welcomeTime.current)}</time>
          </div>
        </div>
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="message-bubble">
              <span>{m.content}</span>
              <time className="message-time">{fmtTime(m.timestamp)}</time>
            </div>
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
          placeholder={messages.length === 0 ? 'Ask about the menu, hours, or dietary info…' : ''}
          ref={inputRef}
          disabled={loading || !sessionId}
          autoFocus
        />
        <button onClick={sendMessage} disabled={loading || !input.trim() || !sessionId}>
          Send
        </button>
      </div>
    </div>
  )
}
