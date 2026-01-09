import { useState } from 'react'
import { useApi, postApi } from '../hooks/useApi'
import { Shield, Search, Upload, AlertTriangle, ExternalLink } from 'lucide-react'

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
    <div className="p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Threat Monitor</h1>
        <p className="text-gray-400">Real-time threat detection and analysis</p>
      </header>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="col-span-2 bg-sentinel-card border border-sentinel-border rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Search className="w-5 h-5 text-cyan-400" />
            Analyze Domain
          </h3>
          <div className="flex gap-4">
            <input
              type="text"
              value={searchDomain}
              onChange={(e) => setSearchDomain(e.target.value)}
              placeholder="Enter domain to analyze..."
              className="flex-1 bg-sentinel-bg border border-sentinel-border rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
              onKeyDown={(e) => e.key === 'Enter' && analyzeDomain()}
            />
            <button
              onClick={analyzeDomain}
              disabled={analyzing}
              className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              {analyzing ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>

          {analysisResult && (
            <div className={`mt-4 p-4 rounded-lg border ${
              analysisResult.is_suspicious 
                ? 'bg-red-500/10 border-red-500/30' 
                : 'bg-green-500/10 border-green-500/30'
            }`}>
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono text-white">{analysisResult.domain}</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  analysisResult.is_suspicious ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                }`}>
                  {analysisResult.is_suspicious ? 'SUSPICIOUS' : 'CLEAN'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Confidence</span>
                  <p className="text-white font-mono">{(analysisResult.confidence * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span className="text-gray-400">DGA Score</span>
                  <p className="text-white font-mono">{(analysisResult.dga_score * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span className="text-gray-400">Heuristic</span>
                  <p className="text-white font-mono">{(analysisResult.heuristic_score * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5 text-cyan-400" />
            Upload PCAP
          </h3>
          <div className="border-2 border-dashed border-sentinel-border rounded-lg p-8 text-center hover:border-cyan-500/50 transition-colors cursor-pointer">
            <Upload className="w-12 h-12 text-gray-500 mx-auto mb-3" />
            <p className="text-gray-400 text-sm">Drop PCAP file here or click to upload</p>
          </div>
        </div>
      </div>

      <div className="bg-sentinel-card border border-sentinel-border rounded-xl overflow-hidden">
        <div className="p-4 border-b border-sentinel-border flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            Detected Threats
          </h3>
          <span className="text-sm text-gray-400">{threats?.length ?? 0} threats</span>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-sentinel-bg/50">
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Domain</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Type</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Confidence</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">DGA Score</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Time</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sentinel-border">
              {(threats ?? []).map((threat) => (
                <tr key={threat.id} className="hover:bg-white/5">
                  <td className="px-4 py-3">
                    <span className="font-mono text-white">{threat.domain}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium uppercase ${
                      threat.threat_type === 'dga' ? 'bg-red-500/20 text-red-400' :
                      threat.threat_type === 'c2' ? 'bg-orange-500/20 text-orange-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {threat.threat_type}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full ${
                            threat.confidence > 0.9 ? 'bg-red-500' :
                            threat.confidence > 0.7 ? 'bg-orange-500' : 'bg-yellow-500'
                          }`}
                          style={{ width: `${threat.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-400 font-mono">
                        {(threat.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-gray-300 font-mono text-sm">
                    {(threat.dga_score * 100).toFixed(0)}%
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-sm">
                    {new Date(threat.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="px-4 py-3">
                    {threat.reported_to_blockchain ? (
                      <span className="text-green-400 text-sm flex items-center gap-1">
                        <Shield className="w-4 h-4" /> Reported
                      </span>
                    ) : (
                      <button
                        onClick={() => reportToBlockchain(threat.id)}
                        className="text-cyan-400 hover:text-cyan-300 text-sm flex items-center gap-1"
                      >
                        <ExternalLink className="w-4 h-4" /> Report
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {(!threats || threats.length === 0) && (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                    No threats detected yet
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

