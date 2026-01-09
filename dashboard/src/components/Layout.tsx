import { Outlet, NavLink } from 'react-router-dom'
import { Shield, Activity, Network, Database, Settings } from 'lucide-react'

const navItems = [
  { to: '/', icon: Activity, label: 'Dashboard' },
  { to: '/threats', icon: Shield, label: 'Threats' },
  { to: '/graph', icon: Network, label: 'Graph' },
  { to: '/blockchain', icon: Database, label: 'Blockchain' },
]

export default function Layout() {
  return (
    <div className="min-h-screen bg-sentinel-bg flex">
      <aside className="w-64 border-r border-sentinel-border bg-sentinel-card/50 flex flex-col">
        <div className="p-6 border-b border-sentinel-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg text-white tracking-tight">SENTINEL</h1>
              <p className="text-xs text-gray-500 font-mono">v1.0.0</p>
            </div>
          </div>
        </div>
        
        <nav className="flex-1 p-4">
          <ul className="space-y-1">
            {navItems.map(({ to, icon: Icon, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                      isActive
                        ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                        : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                    }`
                  }
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        
        <div className="p-4 border-t border-sentinel-border">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>System Active</span>
          </div>
        </div>
      </aside>
      
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}

