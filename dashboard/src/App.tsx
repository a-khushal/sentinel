import React, { useState, useEffect, createContext } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ThreatMonitor from './pages/ThreatMonitor'
import GraphView from './pages/GraphView'
import Blockchain from './pages/Blockchain'
import Federation from './pages/Federation'
import ModelMetrics from './pages/ModelMetrics'

export const ThemeContext = createContext<{
  dark: boolean
  toggle: () => void
}>({ dark: false, toggle: () => {} })

function App() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved === 'dark'
  })

  useEffect(() => {
    document.body.className = dark ? 'dark' : 'light'
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  const toggle = () => setDark(!dark)

  return (
    <ThemeContext.Provider value={{ dark, toggle }}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="threats" element={<ThreatMonitor />} />
            <Route path="graph" element={<GraphView />} />
            <Route path="blockchain" element={<Blockchain />} />
            <Route path="federation" element={<Federation />} />
            <Route path="metrics" element={<ModelMetrics />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeContext.Provider>
  )
}

export default App
