import { useState } from 'react'
import { useApi, postApi } from '../hooks/useApi'
import { Database, Search, Shield, Clock, Hash, Users } from 'lucide-react'

interface BlockchainStatus {
  connected: boolean
  node_registered: boolean
  total_reports: number
}

interface Reputation {
  domain: string
  total_reports: number
  malicious_score: number
  first_seen: number
  last_seen: number
  reporter_count: number
  is_malicious: boolean
}

export default function Blockchain() {
  const { data: status } = useApi<BlockchainStatus>('/blockchain/status', 5000)
  const [searchDomain, setSearchDomain] = useState('')
  const [reputation, setReputation] = useState<Reputation | null>(null)
  const [searching, setSearching] = useState(false)

  const searchReputation = async () => {
    if (!searchDomain) return
    setSearching(true)
    try {
      const response = await fetch(`/api/v1/blockchain/reputation/${encodeURIComponent(searchDomain)}`)
      const data = await response.json()
      setReputation(data)
    } catch (err) {
      console.error('Search failed:', err)
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Blockchain Explorer</h1>
        <p className="text-gray-400">Decentralized threat intelligence ledger</p>
      </header>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <StatusCard
          icon={Database}
          label="Connection Status"
          value={status?.connected ? 'Connected' : 'Disconnected'}
          status={status?.connected ? 'success' : 'error'}
        />
        <StatusCard
          icon={Shield}
          label="Node Status"
          value={status?.node_registered ? 'Registered' : 'Not Registered'}
          status={status?.node_registered ? 'success' : 'warning'}
        />
        <StatusCard
          icon={Hash}
          label="Total Reports"
          value={status?.total_reports?.toString() ?? '0'}
          status="info"
        />
      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Search className="w-5 h-5 text-cyan-400" />
            Query Domain Reputation
          </h3>
          <div className="flex gap-4 mb-4">
            <input
              type="text"
              value={searchDomain}
              onChange={(e) => setSearchDomain(e.target.value)}
              placeholder="Enter domain to check..."
              className="flex-1 bg-sentinel-bg border border-sentinel-border rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
              onKeyDown={(e) => e.key === 'Enter' && searchReputation()}
            />
            <button
              onClick={searchReputation}
              disabled={searching}
              className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              {searching ? 'Searching...' : 'Search'}
            </button>
          </div>

          {reputation && (
            <div className={`p-4 rounded-lg border ${
              reputation.is_malicious 
                ? 'bg-red-500/10 border-red-500/30' 
                : 'bg-green-500/10 border-green-500/30'
            }`}>
              <div className="flex items-center justify-between mb-4">
                <span className="font-mono text-white">{reputation.domain}</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  reputation.is_malicious ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                }`}>
                  {reputation.is_malicious ? 'MALICIOUS' : 'CLEAN'}
                </span>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <ReputationStat icon={Hash} label="Total Reports" value={reputation.total_reports.toString()} />
                <ReputationStat icon={Shield} label="Malicious Score" value={reputation.malicious_score.toString()} />
                <ReputationStat icon={Users} label="Reporters" value={reputation.reporter_count.toString()} />
                <ReputationStat 
                  icon={Clock} 
                  label="First Seen" 
                  value={reputation.first_seen ? new Date(reputation.first_seen * 1000).toLocaleDateString() : 'Never'} 
                />
              </div>
            </div>
          )}
        </div>

        <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Network Info</h3>
          <div className="space-y-4">
            <InfoRow label="Network" value="Polygon Mumbai (Testnet)" />
            <InfoRow label="Contract" value="0x..." truncate />
            <InfoRow label="Min Stake" value="0.01 MATIC" />
            <InfoRow label="Consensus" value="Stake-Weighted Voting" />
          </div>
          
          <div className="mt-6 pt-6 border-t border-sentinel-border">
            <h4 className="text-sm font-medium text-gray-400 mb-3">Quick Actions</h4>
            <div className="flex gap-3">
              <button className="flex-1 py-2 px-4 bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 rounded-lg hover:bg-cyan-600/30 transition-colors text-sm">
                Register Node
              </button>
              <button className="flex-1 py-2 px-4 bg-purple-600/20 text-purple-400 border border-purple-500/30 rounded-lg hover:bg-purple-600/30 transition-colors text-sm">
                View Proposals
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-sentinel-card border border-sentinel-border rounded-xl overflow-hidden">
        <div className="p-4 border-b border-sentinel-border">
          <h3 className="text-lg font-semibold text-white">Recent Transactions</h3>
        </div>
        <div className="p-8 text-center text-gray-500">
          <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No recent transactions</p>
          <p className="text-sm">Transactions will appear here when threats are reported</p>
        </div>
      </div>
    </div>
  )
}

function StatusCard({ icon: Icon, label, value, status }: {
  icon: React.ElementType
  label: string
  value: string
  status: 'success' | 'warning' | 'error' | 'info'
}) {
  const statusColors = {
    success: 'border-green-500/30 bg-green-500/10',
    warning: 'border-yellow-500/30 bg-yellow-500/10',
    error: 'border-red-500/30 bg-red-500/10',
    info: 'border-cyan-500/30 bg-cyan-500/10',
  }

  const iconColors = {
    success: 'text-green-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
    info: 'text-cyan-400',
  }

  return (
    <div className={`border rounded-xl p-6 ${statusColors[status]}`}>
      <Icon className={`w-8 h-8 ${iconColors[status]} mb-3`} />
      <div className="text-xl font-bold text-white mb-1">{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  )
}

function ReputationStat({ icon: Icon, label, value }: {
  icon: React.ElementType
  label: string
  value: string
}) {
  return (
    <div className="flex items-center gap-3">
      <Icon className="w-4 h-4 text-gray-500" />
      <div>
        <div className="text-xs text-gray-400">{label}</div>
        <div className="text-white font-mono">{value}</div>
      </div>
    </div>
  )
}

function InfoRow({ label, value, truncate }: { label: string; value: string; truncate?: boolean }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-sentinel-border last:border-0">
      <span className="text-gray-400">{label}</span>
      <span className={`text-white font-mono text-sm ${truncate ? 'truncate max-w-[200px]' : ''}`}>
        {value}
      </span>
    </div>
  )
}

