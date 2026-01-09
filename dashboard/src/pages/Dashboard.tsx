import { useContext } from 'react'
import { useApi } from '../hooks/useApi'
import { ThemeContext } from '../App'

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

export default function Dashboard() {
  const { dark } = useContext(ThemeContext)
  const { data: stats } = useApi<Stats>('/stats', 2000)
  const { data: threatSummary } = useApi<ThreatSummary>('/threats/stats/summary', 5000)

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold mb-1">System Overview</h1>
        <p className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
          Real-time botnet detection and DNS threat analysis
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatCard 
          label="Queries Processed" 
          value={stats?.total_queries_processed ?? 0} 
          dark={dark} 
        />
        <StatCard 
          label="Threats Detected" 
          value={stats?.threats_detected ?? 0} 
          variant="danger"
          dark={dark} 
        />
        <StatCard 
          label="Domains Analyzed" 
          value={stats?.domains_analyzed ?? 0} 
          dark={dark} 
        />
        <StatCard 
          label="Blockchain Reports" 
          value={threatSummary?.reported_to_blockchain ?? 0} 
          variant="info"
          dark={dark} 
        />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="card p-4">
          <h2 className="font-semibold mb-4 pb-2 border-b border-gray-200 dark:border-gray-700">
            System Status
          </h2>
          <div className="space-y-3">
            <StatusRow label="DNS Capture" active={stats?.capture_running ?? false} dark={dark} />
            <StatusRow label="ML Detection" active={stats?.model_loaded ?? false} dark={dark} />
            <StatusRow label="Graph Analysis" active={true} dark={dark} />
            <StatusRow label="Blockchain Connection" active={true} dark={dark} />
          </div>
        </div>

        <div className="card p-4">
          <h2 className="font-semibold mb-4 pb-2 border-b border-gray-200 dark:border-gray-700">
            Threat Distribution
          </h2>
          <div className="space-y-3">
            {Object.entries(threatSummary?.by_type ?? { dga: 0, c2: 0, tunnel: 0 }).map(([type, count]) => {
              const total = Math.max(threatSummary?.total_threats || 1, 1)
              const pct = ((count as number) / total * 100)
              return (
                <div key={type}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="uppercase font-medium">{type}</span>
                    <span className={`font-mono ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
                      {count} ({pct.toFixed(0)}%)
                    </span>
                  </div>
                  <div className={`h-1.5 rounded-full ${dark ? 'bg-gray-800' : 'bg-gray-200'}`}>
                    <div 
                      className={`h-full rounded-full transition-all ${
                        type === 'dga' ? 'bg-red-500' : 
                        type === 'c2' ? 'bg-orange-500' : 'bg-yellow-500'
                      }`}
                      style={{ width: `${Math.min(pct, 100)}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
          <div className={`mt-4 pt-3 border-t ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex justify-between text-sm">
              <span className={dark ? 'text-gray-400' : 'text-gray-500'}>High Confidence Threats</span>
              <span className="font-mono font-semibold text-red-500">{threatSummary?.high_confidence ?? 0}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className={`px-4 py-3 border-b ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
          <h2 className="font-semibold">Recent Activity</h2>
        </div>
        <table>
          <thead>
            <tr>
              <th style={{ width: '100px' }}>Time</th>
              <th>Event</th>
              <th>Details</th>
              <th style={{ width: '100px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="font-mono text-xs">2s ago</td>
              <td>DGA domain detected</td>
              <td className="font-mono text-xs">xk7h2m9p.evil.com</td>
              <td><span className="badge badge-red">Critical</span></td>
            </tr>
            <tr>
              <td className="font-mono text-xs">15s ago</td>
              <td>Threat reported to blockchain</td>
              <td className="font-mono text-xs">malware.bad</td>
              <td><span className="badge badge-green">Success</span></td>
            </tr>
            <tr>
              <td className="font-mono text-xs">1m ago</td>
              <td>New client connected</td>
              <td className="font-mono text-xs">192.168.1.105</td>
              <td><span className="badge badge-blue">Info</span></td>
            </tr>
            <tr>
              <td className="font-mono text-xs">3m ago</td>
              <td>C2 beacon pattern detected</td>
              <td className="font-mono text-xs">c2server.net</td>
              <td><span className="badge badge-red">Critical</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

function StatCard({ label, value, variant, dark }: { 
  label: string
  value: number
  variant?: 'danger' | 'success' | 'info'
  dark: boolean 
}) {
  const valueColor = variant === 'danger' 
    ? 'text-red-500' 
    : variant === 'success' 
    ? 'text-green-500'
    : variant === 'info'
    ? (dark ? 'text-cyan-400' : 'text-blue-600')
    : ''

  return (
    <div className="card p-4">
      <div className={`text-2xl font-bold font-mono stat-value ${valueColor}`}>
        {value.toLocaleString()}
      </div>
      <div className={`text-xs mt-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
        {label}
      </div>
    </div>
  )
}

function StatusRow({ label, active, dark }: { label: string; active: boolean; dark: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className={`text-sm ${dark ? 'text-gray-300' : 'text-gray-600'}`}>{label}</span>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${active ? 'bg-green-500' : (dark ? 'bg-gray-600' : 'bg-gray-300')}`} />
        <span className={`text-sm ${active ? 'text-green-500' : (dark ? 'text-gray-500' : 'text-gray-400')}`}>
          {active ? 'Active' : 'Inactive'}
        </span>
      </div>
    </div>
  )
}
