import { useState, useEffect, useCallback } from 'react'

const API_BASE = '/api/v1'

export function useApi<T>(endpoint: string, interval?: number) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`)
      if (!response.ok) throw new Error('Failed to fetch')
      const json = await response.json()
      setData(json)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [endpoint])

  useEffect(() => {
    fetchData()
    
    if (interval) {
      const id = setInterval(fetchData, interval)
      return () => clearInterval(id)
    }
  }, [fetchData, interval])

  return { data, loading, error, refetch: fetchData }
}

export async function postApi<T>(endpoint: string, body?: object): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  
  if (!response.ok) {
    throw new Error('Request failed')
  }
  
  return response.json()
}

