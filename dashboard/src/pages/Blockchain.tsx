import { useState, useContext } from 'react'
import { useApi } from '../hooks/useApi'
import { ThemeContext } from '../App'

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
  const { dark } = useContext(ThemeContext)
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
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold mb-1">Blockchain Explorer</h1>
        <p className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
          Decentralized threat intelligence ledger
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Connection Status</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status?.connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className={`font-semibold ${status?.connected ? 'text-green-500' : 'text-red-500'}`}>
              {status?.connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Node Status</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status?.node_registered ? 'bg-green-500' : (dark ? 'bg-gray-600' : 'bg-gray-300')}`} />
            <span className={status?.node_registered ? 'text-green-500 font-semibold' : (dark ? 'text-gray-400' : 'text-gray-500')}>
              {status?.node_registered ? 'Registered' : 'Not Registered'}
            </span>
          </div>
        </div>
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Total Reports</div>
          <div className={`text-xl font-bold font-mono ${dark ? 'text-cyan-400' : 'text-blue-600'}`}>
            {status?.total_reports ?? 0}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="card p-4">
          <h2 className="font-semibold mb-3">Query Domain Reputation</h2>
          <div className="flex gap-3 mb-4">
            <input
              type="text"
              value={searchDomain}
              onChange={(e) => setSearchDomain(e.target.value)}
              placeholder="Enter domain..."
              className="flex-1"
              onKeyDown={(e) => e.key === 'Enter' && searchReputation()}
            />
            <button onClick={searchReputation} disabled={searching} className="btn-primary">
              {searching ? '...' : 'Search'}
            </button>
          </div>

          {reputation && (
            <div className={`p-4 rounded ${
              reputation.is_malicious 
                ? (dark ? 'bg-red-900/20 border border-red-800' : 'bg-red-50 border border-red-200')
                : (dark ? 'bg-green-900/20 border border-green-800' : 'bg-green-50 border border-green-200')
            }`}>
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono font-medium">{reputation.domain}</span>
                <span className={`badge ${reputation.is_malicious ? 'badge-red' : 'badge-green'}`}>
                  {reputation.is_malicious ? 'Malicious' : 'Clean'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Reports</div>
                  <div className="font-mono font-semibold">{reputation.total_reports}</div>
                </div>
                <div>
                  <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Score</div>
                  <div className="font-mono font-semibold">{reputation.malicious_score}</div>
                </div>
                <div>
                  <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Reporters</div>
                  <div className="font-mono font-semibold">{reputation.reporter_count}</div>
                </div>
                <div>
                  <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>First Seen</div>
                  <div className="font-mono font-semibold">
                    {reputation.first_seen ? new Date(reputation.first_seen * 1000).toLocaleDateString() : '-'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="card p-4">
          <h2 className="font-semibold mb-3">Network Information</h2>
          <div className="space-y-2 text-sm">
            <InfoRow label="Network" value="Polygon Mumbai (Testnet)" dark={dark} />
            <InfoRow label="Contract" value="0x..." mono dark={dark} />
            <InfoRow label="Min Stake" value="0.01 MATIC" dark={dark} />
            <InfoRow label="Consensus" value="Stake-Weighted Voting" dark={dark} />
          </div>

          <div className={`mt-4 pt-4 border-t ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex gap-2">
              <button className="flex-1">Register Node</button>
              <button className="flex-1">View Proposals</button>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className={`px-4 py-3 border-b ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
          <h2 className="font-semibold">Recent Transactions</h2>
        </div>
        <div className={`p-8 text-center ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
          <div className="text-3xl mb-2">~</div>
          <p>No recent transactions</p>
          <p className="text-sm mt-1">Transactions will appear when threats are reported</p>
        </div>
      </div>
    </div>
  )
}

function InfoRow({ label, value, mono, dark }: { label: string; value: string; mono?: boolean; dark: boolean }) {
  return (
    <div className={`flex justify-between py-1.5 border-b ${dark ? 'border-gray-700' : 'border-gray-100'} last:border-0`}>
      <span className={dark ? 'text-gray-400' : 'text-gray-500'}>{label}</span>
      <span className={mono ? 'font-mono text-xs' : ''}>{value}</span>
    </div>
  )
}
