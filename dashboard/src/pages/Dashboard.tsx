import { useApi } from '../hooks/useApi'
import { Shield, Activity, Database, AlertTriangle, TrendingUp, Zap } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, AreaChart, Area } from 'recharts'

interface Stats {
  total_queries_processed: number
  threats_detected: number
  domains_analyzed: number
  capture_running: boolean
  model_loaded: boolean
}

interface ThreatSummary {
  total_threats: number
  by_type: Record<string, number>
  high_confidence: number
  reported_to_blockchain: number
}

const mockChartData = Array.from({ length: 24 }, (_, i) => ({
  hour: i,
  queries: Math.floor(Math.random() * 1000) + 500,
  threats: Math.floor(Math.random() * 20),
}))

export default function Dashboard() {
  const { data: stats } = useApi<Stats>('/stats', 2000)
  const { data: threatSummary } = useApi<ThreatSummary>('/threats/stats/summary', 5000)

  return (
    <div className="p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">System Overview</h1>
        <p className="text-gray-400">Real-time botnet detection and DNS threat analysis</p>
      </header>

      <div className="grid grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={Activity}
          label="Queries Processed"
          value={stats?.total_queries_processed ?? 0}
          trend="+12%"
          color="cyan"
        />
        <StatCard
          icon={AlertTriangle}
          label="Threats Detected"
          value={stats?.threats_detected ?? 0}
          trend={threatSummary?.high_confidence ? `${threatSummary.high_confidence} critical` : ''}
          color="red"
        />
        <StatCard
          icon={Shield}
          label="Domains Analyzed"
          value={stats?.domains_analyzed ?? 0}
          trend=""
          color="green"
        />
        <StatCard
          icon={Database}
          label="Blockchain Reports"
          value={threatSummary?.reported_to_blockchain ?? 0}
          trend=""
          color="purple"
        />
      </div>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="col-span-2 bg-sentinel-card border border-sentinel-border rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            Query Volume (24h)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockChartData}>
                <defs>
                  <linearGradient id="queryGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="hour" stroke="#4b5563" fontSize={12} />
                <YAxis stroke="#4b5563" fontSize={12} />
                <Area
                  type="monotone"
                  dataKey="queries"
                  stroke="#06b6d4"
                  fill="url(#queryGradient)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Threat Distribution
          </h3>
          <div className="space-y-4">
            {Object.entries(threatSummary?.by_type ?? { dga: 45, c2: 12, tunnel: 3 }).map(([type, count]) => (
              <div key={type}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400 uppercase">{type}</span>
                  <span className="text-white font-mono">{count}</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      type === 'dga' ? 'bg-red-500' :
                      type === 'c2' ? 'bg-orange-500' : 'bg-yellow-500'
                    }`}
                    style={{ width: `${Math.min((count as number / 60) * 100, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 pt-6 border-t border-sentinel-border">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">High Confidence</span>
              <span className="text-red-400 font-bold">{threatSummary?.high_confidence ?? 0}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <SystemStatus stats={stats} />
        <RecentActivity />
      </div>
    </div>
  )
}

function StatCard({ icon: Icon, label, value, trend, color }: {
  icon: React.ElementType
  label: string
  value: number
  trend: string
  color: 'cyan' | 'red' | 'green' | 'purple'
}) {
  const colors = {
    cyan: 'from-cyan-500/20 to-cyan-500/5 border-cyan-500/30 text-cyan-400',
    red: 'from-red-500/20 to-red-500/5 border-red-500/30 text-red-400',
    green: 'from-green-500/20 to-green-500/5 border-green-500/30 text-green-400',
    purple: 'from-purple-500/20 to-purple-500/5 border-purple-500/30 text-purple-400',
  }

  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-6`}>
      <div className="flex items-center justify-between mb-4">
        <Icon className="w-8 h-8" />
        {trend && <span className="text-xs font-mono opacity-70">{trend}</span>}
      </div>
      <div className="text-3xl font-bold text-white mb-1 font-mono">
        {value.toLocaleString()}
      </div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  )
}

function SystemStatus({ stats }: { stats: Stats | null }) {
  const systems = [
    { name: 'DNS Capture', active: stats?.capture_running ?? false },
    { name: 'ML Detection', active: stats?.model_loaded ?? false },
    { name: 'Graph Analysis', active: true },
    { name: 'Blockchain', active: true },
  ]

  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
      <div className="space-y-3">
        {systems.map(({ name, active }) => (
          <div key={name} className="flex items-center justify-between py-2 border-b border-sentinel-border last:border-0">
            <span className="text-gray-300">{name}</span>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${active ? 'bg-green-500' : 'bg-gray-600'}`} />
              <span className={`text-sm ${active ? 'text-green-400' : 'text-gray-500'}`}>
                {active ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function RecentActivity() {
  const activities = [
    { time: '2s ago', event: 'DGA domain detected', domain: 'xk7h2m9p.evil.com', severity: 'high' },
    { time: '15s ago', event: 'Threat reported to blockchain', domain: 'malware.bad', severity: 'medium' },
    { time: '1m ago', event: 'New client connected', domain: '192.168.1.105', severity: 'low' },
    { time: '3m ago', event: 'C2 beacon pattern detected', domain: 'c2server.net', severity: 'high' },
  ]

  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
      <div className="space-y-3">
        {activities.map((activity, i) => (
          <div key={i} className="flex items-start gap-3 py-2 border-b border-sentinel-border last:border-0">
            <div className={`w-2 h-2 rounded-full mt-2 ${
              activity.severity === 'high' ? 'bg-red-500' :
              activity.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
            }`} />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-300">{activity.event}</p>
              <p className="text-xs text-gray-500 font-mono truncate">{activity.domain}</p>
            </div>
            <span className="text-xs text-gray-500">{activity.time}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

