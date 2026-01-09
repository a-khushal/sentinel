import { Outlet, NavLink } from 'react-router-dom'
import { useContext } from 'react'
import { ThemeContext } from '../App'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/threats', label: 'Threats' },
  { to: '/graph', label: 'Graph' },
  { to: '/blockchain', label: 'Blockchain' },
]

export default function Layout() {
  const { dark, toggle } = useContext(ThemeContext)

  return (
    <div className={`min-h-screen flex flex-col ${dark ? 'bg-[#121212] text-gray-200' : 'bg-[#f7f7f7] text-gray-800'}`}>
      <header className={`border-b ${dark ? 'border-[#2a2a2a] bg-[#1a1a1a]' : 'border-[#e0e0e0] bg-white'}`}>
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${dark ? 'bg-cyan-400' : 'bg-blue-600'}`}></div>
                <span className="font-bold text-base tracking-tight">SENTINEL</span>
              </div>
              <nav className="flex">
                {navItems.map(({ to, label }) => (
                  <NavLink
                    key={to}
                    to={to}
                    className={({ isActive }) =>
                      `px-4 py-4 text-sm border-b-2 -mb-[1px] transition-colors ${
                        isActive
                          ? dark 
                            ? 'text-white border-cyan-400 font-semibold' 
                            : 'text-blue-600 border-blue-600 font-semibold'
                          : dark 
                            ? 'text-gray-400 border-transparent hover:text-gray-200 hover:border-gray-600' 
                            : 'text-gray-500 border-transparent hover:text-gray-800 hover:border-gray-300'
                      }`
                    }
                  >
                    {label}
                  </NavLink>
                ))}
              </nav>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                <span className={`text-xs ${dark ? 'text-gray-400' : 'text-gray-500'}`}>
                  System Active
                </span>
              </div>
              <button
                onClick={toggle}
                className="text-xs"
              >
                {dark ? 'Light Mode' : 'Dark Mode'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
        <Outlet />
      </main>

      <footer className={`border-t py-4 text-center text-xs mt-auto ${
        dark ? 'border-[#2a2a2a] text-gray-600' : 'border-[#e0e0e0] text-gray-400'
      }`}>
        SENTINEL v1.0.0 &middot; Decentralized Botnet Detection System
      </footer>
    </div>
  )
}
