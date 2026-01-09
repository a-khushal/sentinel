import { useState, useContext } from 'react'
import { useApi, postApi } from '../hooks/useApi'
import { ThemeContext } from '../App'

interface FederationStatus {
  active: boolean
  current_round: int
  total_rounds: int
  connected_clients: int
  global_model_hash: string
  privacy_budget_used: number
  last_aggregation: string | null
}

interface FederationHistory {
  rounds: Array<{
    round: number
    clients: number
    samples: number
    loss: number
    model_hash: string
    timestamp: string
  }>
  total_privacy_budget: number
}

interface Client {
  id: string
  status: string
  last_update: string | null
  contributions: number
}

export default function Federation() {
  const { dark } = useContext(ThemeContext)
  const { data: status, refetch } = useApi<FederationStatus>('/federation/status', 2000)
  const { data: history } = useApi<FederationHistory>('/federation/history', 3000)
  const { data: clientsData } = useApi<{clients: Client[], count: number}>('/federation/clients', 3000)

  const [numClients, setNumClients] = useState(3)
  const [rounds, setRounds] = useState(5)
  const [localEpochs, setLocalEpochs] = useState(2)
  const [epsilon, setEpsilon] = useState(1.0)
  const [starting, setStarting] = useState(false)

  const startFederation = async () => {
    setStarting(true)
    try {
      await postApi('/federation/start', {
        num_clients: numClients,
        rounds: rounds,
        local_epochs: localEpochs,
        dp_epsilon: epsilon,
        learning_rate: 0.01
      })
      refetch()
    } catch (err) {
      console.error('Failed to start federation:', err)
    } finally {
      setStarting(false)
    }
  }

  const stopFederation = async () => {
    try {
      await postApi('/federation/stop', {})
      refetch()
    } catch (err) {
      console.error('Failed to stop:', err)
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold mb-1">Federated Learning</h1>
        <p className={`text-sm ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
          Privacy-preserving collaborative model training
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Status</div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status?.active ? 'bg-green-500 animate-pulse' : (dark ? 'bg-gray-600' : 'bg-gray-300')}`} />
            <span className={status?.active ? 'text-green-500 font-semibold' : ''}>
              {status?.active ? 'Training' : 'Idle'}
            </span>
          </div>
        </div>
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Round</div>
          <div className={`text-xl font-bold font-mono ${dark ? 'text-cyan-400' : 'text-blue-600'}`}>
            {status?.current_round ?? 0} / {status?.total_rounds ?? 0}
          </div>
        </div>
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Clients</div>
          <div className={`text-xl font-bold font-mono ${dark ? 'text-cyan-400' : 'text-blue-600'}`}>
            {status?.connected_clients ?? 0}
          </div>
        </div>
        <div className="card p-4">
          <div className={`text-xs mb-1 ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Privacy Budget</div>
          <div className="flex items-center gap-2">
            <div className={`flex-1 h-2 rounded-full ${dark ? 'bg-gray-700' : 'bg-gray-200'}`}>
              <div 
                className="h-full rounded-full bg-purple-500"
                style={{ width: `${Math.min((status?.privacy_budget_used ?? 0) / epsilon * 100, 100)}%` }}
              />
            </div>
            <span className="font-mono text-xs">{(status?.privacy_budget_used ?? 0).toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="card p-4">
          <h2 className="font-semibold mb-4">Training Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Number of Clients</label>
              <input
                type="number"
                value={numClients}
                onChange={(e) => setNumClients(parseInt(e.target.value) || 3)}
                min={2}
                max={10}
                className="w-full mt-1"
                disabled={status?.active}
              />
            </div>
            <div>
              <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Rounds</label>
              <input
                type="number"
                value={rounds}
                onChange={(e) => setRounds(parseInt(e.target.value) || 5)}
                min={1}
                max={50}
                className="w-full mt-1"
                disabled={status?.active}
              />
            </div>
            <div>
              <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Local Epochs</label>
              <input
                type="number"
                value={localEpochs}
                onChange={(e) => setLocalEpochs(parseInt(e.target.value) || 2)}
                min={1}
                max={10}
                className="w-full mt-1"
                disabled={status?.active}
              />
            </div>
            <div>
              <label className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Privacy Epsilon (lower = more private)</label>
              <input
                type="number"
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value) || 1.0)}
                min={0.1}
                max={10}
                step={0.1}
                className="w-full mt-1"
                disabled={status?.active}
              />
            </div>
            <div className="flex gap-3 pt-2">
              {!status?.active ? (
                <button
                  onClick={startFederation}
                  disabled={starting}
                  className="btn-primary flex-1"
                >
                  {starting ? 'Starting...' : 'Start Federation'}
                </button>
              ) : (
                <button
                  onClick={stopFederation}
                  className="btn-danger flex-1"
                >
                  Stop Training
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="card p-4">
          <h2 className="font-semibold mb-4">Connected Nodes</h2>
          {clientsData?.clients && clientsData.clients.length > 0 ? (
            <div className="space-y-2">
              {clientsData.clients.map((client) => (
                <div 
                  key={client.id}
                  className={`p-3 rounded ${dark ? 'bg-gray-800' : 'bg-gray-50'}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${
                        client.status === 'training' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                      }`} />
                      <span className="font-mono text-sm">{client.id}</span>
                    </div>
                    <span className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
                      {client.contributions} rounds
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className={`text-center py-8 ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
              <p>No nodes connected</p>
              <p className="text-xs mt-1">Start federation to simulate nodes</p>
            </div>
          )}
        </div>
      </div>

      <div className="card p-4 mb-6">
        <h2 className="font-semibold mb-4">Privacy Guarantees</h2>
        <div className="grid grid-cols-4 gap-4 text-sm">
          <div>
            <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Mechanism</div>
            <div className="font-semibold">Gaussian DP</div>
          </div>
          <div>
            <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Target Epsilon</div>
            <div className="font-mono font-semibold">{epsilon}</div>
          </div>
          <div>
            <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Delta</div>
            <div className="font-mono font-semibold">10^-5</div>
          </div>
          <div>
            <div className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>Composition</div>
            <div className="font-semibold">Moments Accountant</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className={`px-4 py-3 border-b flex items-center justify-between ${dark ? 'border-gray-700' : 'border-gray-200'}`}>
          <h2 className="font-semibold">Training History</h2>
          {status?.global_model_hash && status.global_model_hash !== 'none' && (
            <span className={`text-xs font-mono ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
              Model: {status.global_model_hash.slice(0, 12)}...
            </span>
          )}
        </div>
        <div className="overflow-x-auto">
          <table>
            <thead>
              <tr>
                <th style={{ width: '80px' }}>Round</th>
                <th style={{ width: '80px' }}>Clients</th>
                <th style={{ width: '100px' }}>Samples</th>
                <th style={{ width: '100px' }}>Loss</th>
                <th>Model Hash</th>
                <th style={{ width: '150px' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {(history?.rounds ?? []).map((round) => (
                <tr key={round.round}>
                  <td className="font-mono">{round.round}</td>
                  <td className="font-mono">{round.clients}</td>
                  <td className="font-mono">{round.samples}</td>
                  <td className="font-mono">{round.loss.toFixed(4)}</td>
                  <td className="font-mono text-xs">{round.model_hash}</td>
                  <td className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
                    {round.timestamp}
                  </td>
                </tr>
              ))}
              {(!history?.rounds || history.rounds.length === 0) && (
                <tr>
                  <td colSpan={6} className={`text-center py-8 ${dark ? 'text-gray-500' : 'text-gray-400'}`}>
                    No training history yet. Start a federation round to begin.
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

