import { useState, useEffect, useRef, useCallback } from 'react'

interface WSMessage {
  type: string
  data: unknown
}

export function useWebSocket(url: string) {
  const [messages, setMessages] = useState<WSMessage[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  const connect = useCallback(() => {
    const ws = new WebSocket(url)
    
    ws.onopen = () => {
      setConnected(true)
    }
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        setMessages(prev => [message, ...prev.slice(0, 99)])
      } catch {
        console.error('Invalid message')
      }
    }
    
    ws.onclose = () => {
      setConnected(false)
      setTimeout(connect, 3000)
    }
    
    ws.onerror = () => {
      ws.close()
    }
    
    wsRef.current = ws
  }, [url])

  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
    }
  }, [connect])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return { messages, connected, clearMessages }
}

