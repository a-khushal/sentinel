import { useState, useContext } from 'react'
import { useApi, postApi } from '../hooks/useApi'
import { ThemeContext } from '../App'

interface Threat {
  id: string
  domain: string
  threat_type: string
  confidence: number
  dga_score: number
  gnn_score: number
  timestamp: string
  reported_to_blockchain: boolean
  tx_hash?: string
}

export default function ThreatMonitor() {
  const { dark } = useContext(ThemeContext)
  const { data: threats, refetch } = useApi<Threat[]>('/threats?limit=100', 3000)
  const [searchDomain, setSearchDomain] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any>(null)

  const analyzeDomain = async () => {
    if (!searchDomain) return
    setAnalyzing(true)
    try {
      const result = await postApi('/threats/analyze', { domain: searchDomain })
      setAnalysisResult(result)
    } catch (err) {
      console.error('Analysis failed:', err)
    } finally {
      setAnalyzing(false)
    }
  }

  const reportToBlockchain = async (threatId: string) => {
    try {
      await postApi(`/threats/${threatId}/report`, {})
      refetch()
    } catch (err) {
      console.error('Report failed:', err)
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold mb-1">Threat Monitor</h1>
        <p className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
          Real-time threat detection and domain analysis
        </p>
      </div>

      <div className="card p-4 mb-6">
        <h2 className="font-semibold mb-3">Analyze Domain</h2>
        <div className="flex gap-3 mb-4">
          <input
            type="text"
            value={searchDomain}
            onChange={(e) => setSearchDomain(e.target.value)}
            placeholder="Enter domain to analyze (e.g., suspicious-domain.com)"
            className="flex-1"
            onKeyDown={(e) => e.key === 'Enter' && analyzeDomain()}
          />
          <button onClick={analyzeDomain} disabled={analyzing} className="btn-primary">
            {analyzing ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {analysisResult && (
          <div className={`p-4 rounded ${
            analysisResult.is_suspicious 
              ? (dark ? 'bg-red-900/20 border border-red-800' : 'bg-red-50 border border-red-200')
              : (dark ? 'bg-green-900/20 border border-green-800' : 'bg-green-50 border border-green-200')
          }`}>
            <div className="flex items-center justify-between mb-3">
              <span className="font-mono font-medium">{analysisResult.domain}</span>
              <span className={`badge ${analysisResult.is_suspicious ? 'badge-red' : 'badge-green'}`}>
                {analysisResult.is_suspicious ? 'Suspicious' : 'Clean'}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Confidence</div>
                <div className="font-mono font-semibold">{(analysisResult.confidence * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>DGA Score</div>
                <div className="font-mono font-semibold">{(analysisResult.dga_score * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Heuristic Score</div>
                <div className="font-mono font-semibold">{(analysisResult.heuristic_score * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <div className={`px-4 py-3 border-b flex items-center justify-between ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
          <h2 className="font-semibold">Detected Threats</h2>
          <span className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
            {threats?.length ?? 0} total
          </span>
        </div>
        <div className="overflow-x-auto">
          <table>
            <thead>
              <tr>
                <th>Domain</th>
                <th style={{ width: '80px' }}>Type</th>
                <th style={{ width: '120px' }}>Confidence</th>
                <th style={{ width: '80px' }}>DGA</th>
                <th style={{ width: '100px' }}>Time</th>
                <th style={{ width: '100px' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {(threats ?? []).map((threat) => (
                <tr key={threat.id}>
                  <td className="font-mono text-xs">{threat.domain}</td>
                  <td>
                    <span className={`badge ${
                      threat.threat_type === 'dga' ? 'badge-red' :
                      threat.threat_type === 'c2' ? 'badge-red' : 'badge-gray'
                    }`}>
                      {threat.threat_type}
                    </span>
                  </td>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className={`w-16 h-1.5 rounded-full ${dark ? 'bg-gray-700' : 'bg-gray-200'}`}>
                        <div 
                          className={`h-full rounded-full ${
                            threat.confidence > 0.9 ? 'bg-red-500' :
                            threat.confidence > 0.7 ? 'bg-orange-500' : 'bg-yellow-500'
                          }`}
                          style={{ width: `${threat.confidence * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs">
                        {(threat.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="font-mono text-xs">{(threat.dga_score * 100).toFixed(0)}%</td>
                  <td className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
                    {new Date(threat.timestamp).toLocaleTimeString()}
                  </td>
                  <td>
                    {threat.reported_to_blockchain ? (
                      <span className="badge badge-green">Reported</span>
                    ) : (
                      <button 
                        onClick={() => reportToBlockchain(threat.id)} 
                        className="text-xs"
                      >
                        Report
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {(!threats || threats.length === 0) && (
                <tr>
                  <td colSpan={6} className={`text-center py-8 ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
                    No threats detected yet. Start DNS capture or upload a PCAP file.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
