import { useEffect, useRef, useState, useCallback, useContext } from 'react'
import { useApi } from '../hooks/useApi'
import { ThemeContext } from '../App'
import ForceGraph2D from 'react-force-graph-2d'

interface GraphNode {
  id: string
  type: string
  label: string
  is_suspicious: boolean
  confidence: number
}

interface GraphEdge {
  source: string
  target: string
  type: string
  weight: number
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats: {
    total_nodes: number
    total_edges: number
    client_nodes: number
    domain_nodes: number
    suspicious_nodes: number
  }
}

export default function GraphView() {
  const { dark } = useContext(ThemeContext)
  const { data: graphData, refetch } = useApi<GraphData>('/graph', 5000)
  const graphRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 })

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth - 2,
          height: 480,
        })
      }
    }
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  const nodeColor = useCallback((node: GraphNode) => {
    if (node.is_suspicious) return '#dc2626'
    switch (node.type) {
      case 'client': return dark ? '#38bdf8' : '#2563eb'
      case 'domain': return dark ? '#a78bfa' : '#7c3aed'
      case 'ip': return dark ? '#4ade80' : '#16a34a'
      default: return '#6b7280'
    }
  }, [dark])

  const graphDataFormatted = graphData ? {
    nodes: graphData.nodes.map(n => ({ ...n })),
    links: graphData.edges.map(e => ({
      source: e.source,
      target: e.target,
      value: e.weight,
    })),
  } : { nodes: [], links: [] }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold mb-1">DNS Traffic Graph</h1>
        <p className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
          Visual representation of DNS query relationships
        </p>
      </div>

      <div className="grid grid-cols-5 gap-3 mb-4">
        <StatBox label="Total Nodes" value={graphData?.stats.total_nodes ?? 0} dark={dark} />
        <StatBox label="Total Edges" value={graphData?.stats.total_edges ?? 0} dark={dark} />
        <StatBox label="Clients" value={graphData?.stats.client_nodes ?? 0} color="blue" dark={dark} />
        <StatBox label="Domains" value={graphData?.stats.domain_nodes ?? 0} color="purple" dark={dark} />
        <StatBox label="Suspicious" value={graphData?.stats.suspicious_nodes ?? 0} color="red" dark={dark} />
      </div>

      <div className="card" ref={containerRef}>
        <div className={`px-4 py-3 border-b flex items-center justify-between ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
          <span className="font-semibold">Network Topology</span>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-4 text-xs">
              <Legend color={dark ? '#38bdf8' : '#2563eb'} label="Client" />
              <Legend color={dark ? '#a78bfa' : '#7c3aed'} label="Domain" />
              <Legend color={dark ? '#4ade80' : '#16a34a'} label="IP" />
              <Legend color="#dc2626" label="Suspicious" />
            </div>
            <button onClick={() => refetch()} className="text-xs">
              Refresh
            </button>
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
            nodeLabel={(node: any) => `${node.label} (${node.type})`}
            linkColor={() => dark ? '#404040' : '#d0d0d0'}
            linkWidth={1}
            backgroundColor={dark ? '#1a1a1a' : '#ffffff'}
          />
        ) : (
          <div className={`h-[480px] flex flex-col items-center justify-center ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
            <div className="text-4xl mb-3">~</div>
            <p>No graph data available</p>
            <p className="text-sm mt-1">Upload a PCAP file or start DNS capture</p>
          </div>
        )}
      </div>
    </div>
  )
}

function StatBox({ label, value, color, dark }: { 
  label: string
  value: number
  color?: 'blue' | 'purple' | 'red'
  dark: boolean 
}) {
  const valueColor = color === 'red' 
    ? 'text-red-500' 
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
