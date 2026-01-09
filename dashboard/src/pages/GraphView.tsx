import { useEffect, useRef, useState, useCallback, useContext } from 'react'
import { postApi, useApi } from '../hooks/useApi'
import { ThemeContext } from '../App'
import ForceGraph2D from 'react-force-graph-2d'

interface GraphNode {
  id: string
  type: string
  label: string
  is_infected?: boolean
  is_c2?: boolean
  predicted_infected?: boolean
  infection_score?: number
  query_count?: number
  client_count?: number
}

interface GraphEdge {
  source: string
  target: string
  type: string
}

interface AnalysisResult {
  is_botnet: boolean
  confidence: number
  verdict: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats: {
    total_nodes: number
    total_edges: number
    clients: number
    domains: number
    predicted_infected: number
    actual_infected: number
  }
  ground_truth?: {
    infected_clients: string[]
    c2_domains: string[]
  }
}

interface CaptureStatus {
  capturing: boolean
  queries_captured: number
  duration_seconds: number
  interface: string
}

type TabType = 'simulated' | 'live'

export default function GraphView() {
  const { dark } = useContext(ThemeContext)
  const graphRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 })
  const [activeTab, setActiveTab] = useState<TabType>('simulated')
  
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [simulateBotnet, setSimulateBotnet] = useState(true)

  const [capturing, setCapturing] = useState(false)
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus | null>(null)
  const [networkInterface, setNetworkInterface] = useState('wlp0s20f3')
  const [captureDuration, setCaptureDuration] = useState(10)

  const { data: statusData } = useApi<CaptureStatus>('/capture/status', capturing ? 1000 : 0)

  useEffect(() => {
    if (statusData) {
      setCaptureStatus(statusData)
      if (!statusData.capturing && capturing) {
        setCapturing(false)
      }
    }
  }, [statusData, capturing])

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth - 2,
          height: 380,
        })
      }
    }
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  const runSimulatedAnalysis = async () => {
    setAnalyzing(true)
    setError(null)
    try {
      const data = await postApi<AnalysisResult>(`/graph/analyze?simulate_botnet=${simulateBotnet}`)
      setResult(data)
    } catch (err) {
      setError('Analysis failed. Make sure T-DGNN model is loaded.')
      console.error(err)
    } finally {
      setAnalyzing(false)
    }
  }

  const startCapture = async () => {
    setError(null)
    setResult(null)
    try {
      await postApi('/capture/start', { 
        interface: networkInterface, 
        duration: captureDuration 
      })
      setCapturing(true)
    } catch (err) {
      setError('Failed to start capture. Requires root privileges for packet sniffing.')
      console.error(err)
    }
  }

  const stopCapture = async () => {
    try {
      await postApi('/capture/stop')
      setCapturing(false)
    } catch (err) {
      console.error(err)
    }
  }

  const analyzeCaptured = async () => {
    setAnalyzing(true)
    setError(null)
    try {
      const data = await postApi<AnalysisResult>('/graph/analyze-captured')
      setResult(data)
    } catch (err) {
      setError('Analysis failed. Need captured DNS queries first.')
      console.error(err)
    } finally {
      setAnalyzing(false)
    }
  }

  const nodeColor = useCallback((node: GraphNode) => {
    const score = node.infection_score ?? 0
    
    if (node.predicted_infected || score > 0.5) {
      const intensity = Math.min(score, 1)
      return `rgba(220, 38, 38, ${0.5 + intensity * 0.5})`
    }
    
    if (node.type === 'client') {
      return dark ? '#38bdf8' : '#2563eb'
    }
    if (node.type === 'ip') {
      return dark ? '#4ade80' : '#16a34a'
    }
    return dark ? '#a78bfa' : '#7c3aed'
  }, [dark])

  const nodeSize = useCallback((node: GraphNode) => {
    if (node.predicted_infected) return 8
    return 5
  }, [])

  const graphDataFormatted = result ? {
    nodes: result.nodes.map(n => ({ ...n })),
    links: result.edges.map(e => ({
      source: e.source,
      target: e.target,
    })),
  } : { nodes: [], links: [] }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold mb-1">T-DGNN Graph Analysis</h1>
        <p className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
          Temporal Dynamic Graph Neural Network for botnet detection
        </p>
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={() => { setActiveTab('simulated'); setResult(null); setError(null); }}
          className={`px-4 py-2 text-sm font-medium rounded-t border-b-2 ${
            activeTab === 'simulated'
              ? 'border-blue-500 text-blue-500'
              : dark ? 'border-transparent text-gray-400 hover:text-gray-300' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Simulated Traffic
        </button>
        <button
          onClick={() => { setActiveTab('live'); setResult(null); setError(null); }}
          className={`px-4 py-2 text-sm font-medium rounded-t border-b-2 ${
            activeTab === 'live'
              ? 'border-green-500 text-green-500'
              : dark ? 'border-transparent text-gray-400 hover:text-gray-300' : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Live DNS Capture
        </button>
      </div>

      {activeTab === 'simulated' && (
        <div className="card p-4 mb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={simulateBotnet}
                  onChange={(e) => setSimulateBotnet(e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-sm">Inject botnet traffic</span>
              </label>
            </div>
            <button 
              onClick={runSimulatedAnalysis} 
              disabled={analyzing}
              className="btn-primary"
            >
              {analyzing ? 'Analyzing...' : 'Run Simulation'}
            </button>
          </div>
          <p className={`text-xs mt-2 ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
            Generates fake network with clients querying domains. If checked, some clients will query DGA domains.
          </p>
        </div>
      )}

      {activeTab === 'live' && (
        <div className="card p-4 mb-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-4">
              <div>
                <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Interface</label>
                <input
                  type="text"
                  value={networkInterface}
                  onChange={(e) => setNetworkInterface(e.target.value)}
                  className="block w-24 px-2 py-1 text-sm border rounded bg-transparent"
                  placeholder="eth0"
                />
              </div>
              <div>
                <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Duration (sec)</label>
                <input
                  type="text"
                  value={captureDuration}
                  onChange={(e) => {
                    const val = e.target.value.replace(/\D/g, '')
                    setCaptureDuration(val === '' ? 0 : Math.min(300, parseInt(val)))
                  }}
                  onBlur={() => {
                    if (captureDuration < 5) setCaptureDuration(5)
                  }}
                  className="block w-20 px-2 py-1 text-sm border rounded bg-transparent"
                />
              </div>
            </div>
            <div className="flex gap-2">
              {!capturing ? (
                <button onClick={startCapture} className="btn-primary bg-green-600 hover:bg-green-700">
                  Start Capture
                </button>
              ) : (
                <button onClick={stopCapture} className="btn-primary bg-red-600 hover:bg-red-700">
                  Stop Capture
                </button>
              )}
              <button 
                onClick={analyzeCaptured} 
                disabled={analyzing || capturing}
                className="btn-primary"
              >
                {analyzing ? 'Analyzing...' : 'Analyze Captured'}
              </button>
            </div>
          </div>
          
          {captureStatus && (
            <div className={`flex items-center gap-6 text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
              <span className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${captureStatus.capturing ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
                {captureStatus.capturing ? 'Capturing' : 'Idle'}
              </span>
              <span>Queries: <strong className="text-white">{captureStatus.queries_captured}</strong></span>
              <span>Duration: {captureStatus.duration_seconds}s</span>
            </div>
          )}
          
          <div className={`text-xs mt-3 p-2 rounded ${dark ? 'bg-yellow-900/30 text-yellow-400' : 'bg-yellow-100 text-yellow-700'}`}>
            Requires root/sudo to capture packets. Run backend with: <code className="font-mono">sudo python -m uvicorn api.main:app</code>
          </div>
        </div>
      )}

      {error && (
        <div className="card p-4 mb-4 bg-red-900/20 border border-red-800 text-red-400">
          {error}
        </div>
      )}

      {result && (
        <>
          <div className={`card p-4 mb-4 ${
            result.is_botnet 
              ? 'bg-red-900/20 border border-red-700' 
              : result.verdict === 'Suspicious Activity'
              ? 'bg-yellow-900/20 border border-yellow-700'
              : 'bg-green-900/20 border border-green-700'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <div className={`text-2xl font-bold ${
                  result.is_botnet 
                    ? 'text-red-400' 
                    : result.verdict === 'Suspicious Activity'
                    ? 'text-yellow-400'
                    : 'text-green-400'
                }`}>
                  {result.verdict}
                </div>
                <div className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div className={`text-5xl ${
                result.is_botnet 
                  ? 'text-red-500' 
                  : result.verdict === 'Suspicious Activity'
                  ? 'text-yellow-500'
                  : 'text-green-500'
              }`}>
                {result.is_botnet ? '!' : result.verdict === 'Suspicious Activity' ? '?' : 'OK'}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-6 gap-3 mb-4">
            <StatBox label="Nodes" value={result.stats.total_nodes} dark={dark} />
            <StatBox label="Edges" value={result.stats.total_edges} dark={dark} />
            <StatBox label="Clients" value={result.stats.clients} color="blue" dark={dark} />
            <StatBox label="Domains" value={result.stats.domains} color="purple" dark={dark} />
            <StatBox label="Predicted Infected" value={result.stats.predicted_infected} color="red" dark={dark} />
            {activeTab === 'simulated' && (
              <StatBox label="Actual Infected" value={result.stats.actual_infected} color="orange" dark={dark} />
            )}
            {activeTab === 'live' && (
              <StatBox label="Suspicious" value={result.stats.predicted_infected} color="red" dark={dark} />
            )}
          </div>
        </>
      )}

      <div className="card" ref={containerRef}>
        <div className={`px-4 py-3 border-b flex items-center justify-between ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
          <span className="font-semibold">Network Graph</span>
          <div className="flex items-center gap-4 text-xs">
            <Legend color={dark ? '#38bdf8' : '#2563eb'} label="Client" />
            <Legend color={dark ? '#a78bfa' : '#7c3aed'} label="Domain" />
            <Legend color={dark ? '#4ade80' : '#16a34a'} label="IP" />
            <Legend color="#dc2626" label="Infected" />
          </div>
        </div>
        
        {graphDataFormatted.nodes.length > 0 ? (
          <ForceGraph2D
            ref={graphRef}
            graphData={graphDataFormatted}
            width={dimensions.width}
            height={dimensions.height}
            nodeColor={nodeColor as any}
            nodeRelSize={5}
            nodeVal={nodeSize as any}
            nodeLabel={(node: any) => {
              const score = node.infection_score ? ` [${(node.infection_score * 100).toFixed(0)}%]` : ''
              return `${node.label} (${node.type})${score}`
            }}
            linkColor={() => dark ? '#404040' : '#d0d0d0'}
            linkWidth={1}
            backgroundColor={dark ? '#1a1a1a' : '#ffffff'}
            cooldownTicks={100}
          />
        ) : (
          <div className={`h-[380px] flex flex-col items-center justify-center ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
            <div className="text-4xl mb-3">{activeTab === 'live' ? '~' : '#'}</div>
            <p>No graph data</p>
            <p className="text-sm mt-1">
              {activeTab === 'simulated' 
                ? 'Click "Run Simulation" to generate a network' 
                : 'Capture DNS traffic then click "Analyze Captured"'}
            </p>
          </div>
        )}
      </div>

      {result && result.ground_truth && activeTab === 'simulated' && (
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="card p-4">
            <h3 className="font-semibold mb-2 text-red-400">Infected Clients (Ground Truth)</h3>
            <div className="space-y-1 font-mono text-xs">
              {result.ground_truth.infected_clients.map(c => (
                <div key={c} className={dark ? 'text-gray-300' : 'text-gray-700'}>{c}</div>
              ))}
            </div>
          </div>
          <div className="card p-4">
            <h3 className="font-semibold mb-2 text-red-400">C2 Domains (Ground Truth)</h3>
            <div className="space-y-1 font-mono text-xs max-h-32 overflow-auto">
              {result.ground_truth.c2_domains.map(d => (
                <div key={d} className={dark ? 'text-gray-300' : 'text-gray-700'}>{d}</div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function StatBox({ label, value, color, dark }: { 
  label: string
  value: number
  color?: 'blue' | 'purple' | 'red' | 'orange' | 'green'
  dark: boolean 
}) {
  const valueColor = color === 'red' 
    ? 'text-red-500' 
    : color === 'orange'
    ? 'text-orange-500'
    : color === 'green'
    ? 'text-green-500'
    : color === 'blue' 
    ? (dark ? 'text-sky-400' : 'text-blue-600')
    : color === 'purple'
    ? (dark ? 'text-violet-400' : 'text-violet-600')
    : ''

  return (
    <div className="card p-3 text-center">
      <div className={`text-xl font-bold font-mono ${valueColor}`}>{value}</div>
      <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>{label}</div>
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      <span>{label}</span>
    </div>
  )
}
