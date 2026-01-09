import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ThreatMonitor from './pages/ThreatMonitor'
import GraphView from './pages/GraphView'
import Blockchain from './pages/Blockchain'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="threats" element={<ThreatMonitor />} />
          <Route path="graph" element={<GraphView />} />
          <Route path="blockchain" element={<Blockchain />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App

