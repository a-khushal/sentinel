import { useEffect, useRef, useState, useCallback } from 'react'
import { useApi } from '../hooks/useApi'
import { Network, ZoomIn, ZoomOut, Maximize2, RefreshCw } from 'lucide-react'
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
  const { data: graphData, refetch } = useApi<GraphData>('/graph', 5000)
  const graphRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight - 80,
        })
      }
    }
    
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  const nodeColor = useCallback((node: GraphNode) => {
    if (node.is_suspicious) return '#ef4444'
    switch (node.type) {
      case 'client': return '#06b6d4'
      case 'domain': return '#8b5cf6'
      case 'ip': return '#10b981'
      default: return '#6b7280'
    }
  }, [])

  const nodeSize = useCallback((node: GraphNode) => {
    if (node.is_suspicious) return 8
    return node.type === 'client' ? 6 : 4
  }, [])

  const graphDataFormatted = graphData ? {
    nodes: graphData.nodes.map(n => ({ ...n, id: n.id })),
    links: graphData.edges.map(e => ({
      source: e.source,
      target: e.target,
      type: e.type,
      value: e.weight,
    })),
  } : { nodes: [], links: [] }

  return (
    <div className="p-8 h-screen flex flex-col">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">DNS Traffic Graph</h1>
          <p className="text-gray-400">Visual representation of DNS relationships</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => graphRef.current?.zoomToFit(400)}
            className="p-2 bg-sentinel-card border border-sentinel-border rounded-lg hover:bg-white/5"
          >
            <Maximize2 className="w-5 h-5 text-gray-400" />
          </button>
          <button
            onClick={() => refetch()}
            className="p-2 bg-sentinel-card border border-sentinel-border rounded-lg hover:bg-white/5"
          >
            <RefreshCw className="w-5 h-5 text-gray-400" />
          </button>
        </div>
      </header>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatBox label="Total Nodes" value={graphData?.stats.total_nodes ?? 0} color="cyan" />
        <StatBox label="Total Edges" value={graphData?.stats.total_edges ?? 0} color="purple" />
        <StatBox label="Clients" value={graphData?.stats.client_nodes ?? 0} color="blue" />
        <StatBox label="Suspicious" value={graphData?.stats.suspicious_nodes ?? 0} color="red" />
      </div>

      <div 
        ref={containerRef}
        className="flex-1 bg-sentinel-card border border-sentinel-border rounded-xl overflow-hidden relative"
      >
        {graphDataFormatted.nodes.length > 0 ? (
          <ForceGraph2D
            ref={graphRef}
            graphData={graphDataFormatted}
            width={dimensions.width}
            height={dimensions.height}
            nodeColor={nodeColor as any}
            nodeRelSize={nodeSize as any}
            nodeLabel={(node: any) => `${node.label} (${node.type})`}
            linkColor={() => 'rgba(75, 85, 99, 0.5)'}
            linkWidth={(link: any) => Math.min(link.value, 3)}
            backgroundColor="#111827"
            nodeCanvasObject={(node: any, ctx, globalScale) => {
              const size = nodeSize(node)
              const color = nodeColor(node)
              
              ctx.beginPath()
              ctx.arc(node.x, node.y, size, 0, 2 * Math.PI)
              ctx.fillStyle = color
              ctx.fill()
              
              if (node.is_suspicious) {
                ctx.strokeStyle = '#ef4444'
                ctx.lineWidth = 2
                ctx.stroke()
              }
              
              if (globalScale > 1.5) {
                ctx.font = `${10/globalScale}px JetBrains Mono`
                ctx.textAlign = 'center'
                ctx.textBaseline = 'top'
                ctx.fillStyle = '#9ca3af'
                ctx.fillText(node.label.slice(0, 20), node.x, node.y + size + 2)
              }
            }}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <Network className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No graph data available</p>
              <p className="text-gray-500 text-sm">Upload a PCAP or start DNS capture</p>
            </div>
          </div>
        )}

        <div className="absolute bottom-4 left-4 bg-sentinel-bg/90 backdrop-blur rounded-lg p-3 border border-sentinel-border">
          <div className="text-xs text-gray-400 mb-2">Legend</div>
          <div className="space-y-1">
            <LegendItem color="#06b6d4" label="Client" />
            <LegendItem color="#8b5cf6" label="Domain" />
            <LegendItem color="#10b981" label="IP" />
            <LegendItem color="#ef4444" label="Suspicious" />
          </div>
        </div>
      </div>
    </div>
  )
}

function StatBox({ label, value, color }: { label: string; value: number; color: string }) {
  const colors: Record<string, string> = {
    cyan: 'border-cyan-500/30 text-cyan-400',
    purple: 'border-purple-500/30 text-purple-400',
    blue: 'border-blue-500/30 text-blue-400',
    red: 'border-red-500/30 text-red-400',
  }

  return (
    <div className={`bg-sentinel-card border ${colors[color]} rounded-lg p-4`}>
      <div className="text-2xl font-bold font-mono text-white">{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  )
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-xs text-gray-300">{label}</span>
    </div>
  )
}

