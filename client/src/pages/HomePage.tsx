import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { API_URL, type Restaurant } from '../api'

export default function HomePage() {
  const [restaurants, setRestaurants] = useState<Restaurant[]>([])
  const [error, setError] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    fetch(`${API_URL}/restaurants`)
      .then(r => r.json())
      .then(setRestaurants)
      .catch(() => setError(true))
  }, [])

  return (
    <div className="home-container">
      <header className="home-header">
        <h1>🍽️ Restaurant Assistant</h1>
        <p>Choose a restaurant to start chatting</p>
      </header>

      {error && (
        <p className="home-error">Could not reach the server. Is it running?</p>
      )}

      <div className="restaurant-grid">
        {restaurants.map(r => (
          <button
            key={r.id}
            className="restaurant-card"
            onClick={() => navigate(`/chat/${r.id}`, { state: r })}
          >
            <span className="restaurant-icon">{r.icon}</span>
            <span className="restaurant-name">{r.name}</span>
            <span className="restaurant-desc">{r.description}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
