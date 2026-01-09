import { useState, useContext } from 'react'
import { useApi, postApi } from '../hooks/useApi'
import { ThemeContext } from '../App'

interface BlockchainStatus {
  connected: boolean
  node_registered: boolean
  total_reports: number
  network?: string
  contract?: string
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

interface ReportResult {
  success: boolean
  tx_hash: string
  domain: string
}

export default function Blockchain() {
  const { dark } = useContext(ThemeContext)
  const { data: status, refetch: refetchStatus } = useApi<BlockchainStatus>('/blockchain/status', 5000)
  
  const [searchDomain, setSearchDomain] = useState('')
  const [reputation, setReputation] = useState<Reputation | null>(null)
  const [searching, setSearching] = useState(false)

  const [reportDomain, setReportDomain] = useState('')
  const [reportType, setReportType] = useState('dga')
  const [reportConfidence, setReportConfidence] = useState(85)
  const [reporting, setReporting] = useState(false)
  const [reportResult, setReportResult] = useState<ReportResult | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)

  const [registering, setRegistering] = useState(false)

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

  const reportThreat = async () => {
    if (!reportDomain) return
    setReporting(true)
    setReportError(null)
    setReportResult(null)
    
    try {
      const result = await postApi<ReportResult>('/blockchain/report', {
        domain: reportDomain,
        threat_type: reportType,
        confidence: reportConfidence,
        evidence: `Reported via SENTINEL UI at ${new Date().toISOString()}`
      })
      setReportResult(result)
      setReportDomain('')
      refetchStatus()
    } catch (err) {
      setReportError('Failed to report. Make sure node is registered.')
      console.error(err)
    } finally {
      setReporting(false)
    }
  }

  const registerNode = async () => {
    setRegistering(true)
    try {
      await postApi('/blockchain/register?stake=0.01')
      refetchStatus()
    } catch (err) {
      console.error('Registration failed:', err)
    } finally {
      setRegistering(false)
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold mb-1">Blockchain Threat Ledger</h1>
        <p className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
          Immutable threat intelligence on Ethereum Sepolia
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Network</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status?.connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="font-semibold">
              {status?.connected ? 'Sepolia' : 'Disconnected'}
            </span>
          </div>
        </div>
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Node Status</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status?.node_registered ? 'bg-green-500' : 'bg-yellow-500'}`} />
            <span className={status?.node_registered ? 'text-green-500 font-semibold' : 'text-yellow-500'}>
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
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Chain ID</div>
          <div className="font-mono text-sm">11155111</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="card p-4">
          <h2 className="font-semibold mb-3">Report Threat to Blockchain</h2>
          
          {!status?.node_registered && (
            <div className={`mb-4 p-3 rounded text-sm ${dark ? 'bg-yellow-900/30 text-yellow-400' : 'bg-yellow-100 text-yellow-700'}`}>
              Node not registered. Register first to report threats.
              <button 
                onClick={registerNode}
                disabled={registering}
                className="ml-2 underline"
              >
                {registering ? 'Registering...' : 'Register Now'}
              </button>
            </div>
          )}

          <div className="space-y-3">
            <div>
              <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Domain</label>
              <input
                type="text"
                value={reportDomain}
                onChange={(e) => setReportDomain(e.target.value)}
                placeholder="malicious-domain.xyz"
                className="w-full mt-1"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Threat Type</label>
                <select 
                  value={reportType} 
                  onChange={(e) => setReportType(e.target.value)}
                  className="w-full mt-1"
                >
                  <option value="dga">DGA</option>
                  <option value="c2">C2 Server</option>
                  <option value="tunnel">DNS Tunnel</option>
                  <option value="unknown">Unknown</option>
                </select>
              </div>
              <div>
                <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Confidence (%)</label>
                <input
                  type="number"
                  value={reportConfidence}
                  onChange={(e) => setReportConfidence(parseInt(e.target.value) || 0)}
                  min={1}
                  max={100}
                  className="w-full mt-1"
                />
              </div>
            </div>
            <button 
              onClick={reportThreat}
              disabled={reporting || !status?.node_registered || !reportDomain}
              className="btn-primary w-full"
            >
              {reporting ? 'Submitting to Blockchain...' : 'Report Threat'}
            </button>
          </div>

          {reportError && (
            <div className={`mt-3 p-3 rounded text-sm ${dark ? 'bg-red-900/30 text-red-400' : 'bg-red-100 text-red-700'}`}>
              {reportError}
            </div>
          )}

          {reportResult && (
            <div className={`mt-3 p-3 rounded text-sm ${dark ? 'bg-green-900/30 text-green-400' : 'bg-green-100 text-green-700'}`}>
              <div className="font-semibold mb-1">Reported to blockchain!</div>
              <div className="font-mono text-xs break-all">
                TX: {reportResult.tx_hash.slice(0, 20)}...
              </div>
              <a 
                href={`https://sepolia.etherscan.io/tx/${reportResult.tx_hash}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs underline mt-1 inline-block"
              >
                View on Etherscan
              </a>
            </div>
          )}
        </div>

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

          {!reputation && (
            <div className={`text-center py-8 ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
              <p>Enter a domain to check its reputation</p>
            </div>
          )}
        </div>
      </div>

      <div className="card p-4">
        <h2 className="font-semibold mb-3">Contract Info</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <InfoRow label="Network" value="Ethereum Sepolia (Testnet)" dark={dark} />
          <InfoRow label="Chain ID" value="11155111" mono dark={dark} />
          <InfoRow 
            label="ThreatLedger" 
            value="0xb5430433ba52d853F667BC735fc453e389855E64" 
            mono 
            dark={dark}
            link="https://sepolia.etherscan.io/address/0xb5430433ba52d853F667BC735fc453e389855E64"
          />
          <InfoRow 
            label="FederatedGovernance" 
            value="0xB560339aC4985bAea1764811D8cD4ed46A96C477" 
            mono 
            dark={dark}
            link="https://sepolia.etherscan.io/address/0xB560339aC4985bAea1764811D8cD4ed46A96C477"
          />
        </div>
      </div>
    </div>
  )
}

function InfoRow({ label, value, mono, dark, link }: { 
  label: string
  value: string
  mono?: boolean
  dark: boolean
  link?: string
}) {
  const content = (
    <span className={`${mono ? 'font-mono text-xs' : ''} ${link ? 'underline cursor-pointer' : ''}`}>
      {mono && value.length > 20 ? `${value.slice(0, 10)}...${value.slice(-8)}` : value}
    </span>
  )

  return (
    <div className={`flex justify-between py-1.5 border-b ${dark ? 'border-gray-700' : 'border-gray-100'} last:border-0`}>
      <span className={dark ? 'text-gray-400' : 'text-gray-500'}>{label}</span>
      {link ? (
        <a href={link} target="_blank" rel="noopener noreferrer">{content}</a>
      ) : content}
    </div>
  )
}
