import { useState, useEffect, useRef, useCallback } from 'react'
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, Time, LineStyle } from 'lightweight-charts'

const API = '/api'

interface Strategy {
  name: string
  config: any
}

interface BacktestSummary {
  config_name: string
  k_type: string
  backtest_mode: string
  start_date: string
  end_date: string
  initial_capital: number
  final_capital: number
  total_return_pct: number
  annual_return_pct: number
  max_drawdown_pct: number
  sharpe_ratio: number
  win_rate: number
  profit_loss_ratio: number
  expected_value: number
  total_trades: number
  win_trades: number
  lose_trades: number
  avg_profit_pct: number
  avg_loss_pct: number
  max_profit_pct: number
  max_loss_pct: number
  avg_hold_days: number
}

interface Trade {
  code: string
  buy_date: string
  buy_price: number
  sell_date: string
  sell_price: number
  sell_reason: string
  shares: number
  profit_pct: number
  profit_amount: number
  hold_days: number
  action: string  // "buy" | "add" | "reduce" | "clear"
  trade_id: string  // 唯一交易ID
}

interface EquityPoint {
  date: string
  equity: number
  cash: number
  positions: number
}

interface BacktestResult {
  task_id: string
  status: string
  summary: BacktestSummary
  trades: Trade[]
  equity_curve: EquityPoint[]
  annual_returns: { period: string; return_pct: number }[]
  monthly_returns: { period: string; return_pct: number }[]
}

interface KLinePoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface Annotation {
  date: string
  type: string  // "buy" | "add" | "reduce" | "clear"
  price: number
  reason: string
  profit_pct?: number
  task_id?: string
  strategy?: string
  trade_id?: string  // 唯一交易ID：从建仓到清仓
}

type Tab = 'strategies' | 'backtest' | 'result' | 'kline'

const ACTION_LABELS: Record<string, { label: string; color: string; shape: string }> = {
  buy: { label: '买入', color: '#ef4444', shape: 'arrowUp' },
  add: { label: '加仓', color: '#f97316', shape: 'arrowUp' },
  reduce: { label: '减仓', color: '#22c55e', shape: 'arrowDown' },
  clear: { label: '清仓', color: '#06b6d4', shape: 'arrowDown' },
}

export default function App() {
  const [tab, setTab] = useState<Tab>('strategies')
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [selectedPool, setSelectedPool] = useState<string>('全量')
  const [stockPools, setStockPools] = useState<{name: string; description: string; count: number}[]>([])
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState('2026-06-06')
  const [taskId, setTaskId] = useState<string>('')
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [polling, setPolling] = useState(false)
  const [showCreate, setShowCreate] = useState(false)
  const [editName, setEditName] = useState('')
  const [editYaml, setEditYaml] = useState('')
  const [klineCode, setKlineCode] = useState('')

  useEffect(() => { loadStrategies(); loadStockPools() }, [])

  // 轮询回测进度
  useEffect(() => {
    if (!polling || !taskId) return
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API}/backtest/${taskId}`)
        const data = await res.json()
        if (data.status === 'completed') {
          setResult(data as BacktestResult)
          setPolling(false)
          setTab('result')
        } else if (data.status === 'failed') {
          alert('回测失败: ' + (data.error || '未知错误'))
          setPolling(false)
        }
      } catch {}
    }, 1000)
    return () => clearInterval(timer)
  }, [polling, taskId])

  async function loadStrategies() {
    try {
      const res = await fetch(`${API}/strategies`)
      const data = await res.json()
      setStrategies(data.strategies || [])
    } catch {}
  }

  async function loadStockPools() {
    try {
      const res = await fetch(`${API}/stock-pools`)
      const data = await res.json()
      setStockPools(data.pools || [])
    } catch {}
  }

  async function startBacktest() {
    if (!selectedStrategy) return
    try {
      const res = await fetch(`${API}/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_name: selectedStrategy,
          start_date: startDate || null,
          end_date: endDate || null,
          stock_pool_name: selectedPool,
        }),
      })
      const data = await res.json()
      setTaskId(data.task_id)
      setPolling(true)
    } catch (e) {
      alert('启动回测失败')
    }
  }

  async function deleteStrategy(name: string) {
    if (!confirm(`确定删除策略 "${name}"？`)) return
    await fetch(`${API}/strategies/${name}`, { method: 'DELETE' })
    loadStrategies()
  }

  async function saveStrategy() {
    if (!editName.trim()) return
    try {
      const config = JSON.parse(editYaml)
      await fetch(`${API}/strategies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: editName, config }),
      })
      setShowCreate(false)
      loadStrategies()
    } catch {
      alert('配置格式错误，请输入有效的 JSON')
    }
  }

  function openEdit(name: string) {
    const s = strategies.find(s => s.name === name)
    if (!s) return
    setEditName(name)
    setEditYaml(JSON.stringify(s.config, null, 2))
    setShowCreate(true)
  }

  function openKline(code: string) {
    setKlineCode(code)
    setTab('kline')
  }

  const fmtPct = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%'

  return (
    <div className="app">
      <div className="header">
        <h1>量化回测系统</h1>
        <span style={{ fontSize: 12, color: 'var(--text2)' }}>
          数据覆盖 5062 只A股 | 日K / 周K
        </span>
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button className={`tab ${tab === 'strategies' ? 'active' : ''}`} onClick={() => setTab('strategies')}>
          策略管理
        </button>
        <button className={`tab ${tab === 'backtest' ? 'active' : ''}`} onClick={() => setTab('backtest')}>
          运行回测
        </button>
        <button className={`tab ${tab === 'result' ? 'active' : ''}`} onClick={() => setTab('result')} disabled={!result}>
          回测结果
        </button>
        <button className={`tab ${tab === 'kline' ? 'active' : ''}`} onClick={() => setTab('kline')} disabled={!klineCode}>
          K线标注
        </button>
      </div>

      {/* ===================== 策略管理 ===================== */}
      {tab === 'strategies' && (
        <div>
          <div className="flex-between mb-16">
            <h2 style={{ fontSize: 16 }}>已保存策略</h2>
            <button className="btn btn-primary" onClick={() => { setEditName(''); setEditYaml(''); setShowCreate(true) }}>
              + 新建策略
            </button>
          </div>

          {strategies.length === 0 ? (
            <div className="empty">暂无策略，点击上方按钮创建</div>
          ) : (
            strategies.map(s => (
              <div key={s.name} className="strategy-item">
                <div>
                  <span className="name">{s.name}</span>
                  <span className="ktype">{s.config.k_type || 'daily'}</span>
                  <span className="ktype" style={{ marginLeft: 4, background: s.config.backtest_mode === 'portfolio' ? 'var(--surface)' : '#1a3a2a' }}>
                    {s.config.backtest_mode === 'portfolio' ? 'portfolio' : 'signal'}
                  </span>
                  {(s.config.buy_price_type || s.config.sell_price_type || s.config.buy_execution || s.config.sell_execution) && (
                    <span className="ktype" style={{ marginLeft: 4, background: '#2a2a3a', color: '#a78bfa' }}>
                      买:{s.config.buy_price_type || 'close'}{s.config.buy_execution === 'next_day' ? '+1' : ''} 卖:{s.config.sell_price_type || 'close'}{s.config.sell_execution === 'next_day' ? '+1' : ''}
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn btn-outline" onClick={() => openEdit(s.name)}>编辑</button>
                  <button className="btn btn-danger" onClick={() => deleteStrategy(s.name)}>删除</button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* ===================== 运行回测 ===================== */}
      {tab === 'backtest' && (
        <div>
          <div className="card">
            <h2>运行回测</h2>
            <div className="form-group">
              <label>选择策略</label>
              <select value={selectedStrategy} onChange={e => setSelectedStrategy(e.target.value)}>
                <option value="">-- 请选择策略 --</option>
                {strategies.map(s => (
                  <option key={s.name} value={s.name}>{s.name} ({s.config.k_type || 'daily'})</option>
                ))}
              </select>
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label>开始日期</label>
                <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label>结束日期</label>
                <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
              </div>
            </div>
            <div className="form-group">
              <label>标的池</label>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {stockPools.map(p => (
                  <button
                    key={p.name}
                    onClick={() => setSelectedPool(p.name)}
                    style={{
                      padding: '6px 14px',
                      borderRadius: 6,
                      border: selectedPool === p.name ? '2px solid var(--accent)' : '1px solid var(--border)',
                      background: selectedPool === p.name ? 'var(--accent-bg)' : 'var(--bg2)',
                      color: selectedPool === p.name ? 'var(--accent)' : 'var(--text1)',
                      fontWeight: selectedPool === p.name ? 600 : 400,
                      cursor: 'pointer',
                      fontSize: 13,
                      transition: 'all 0.15s',
                    }}
                  >
                    {p.name}
                    <span style={{ fontSize: 11, marginLeft: 4, opacity: 0.6 }}>({p.count})</span>
                  </button>
                ))}
              </div>
            </div>
            <button className="btn btn-primary" onClick={startBacktest} disabled={!selectedStrategy || polling}>
              {polling ? <><span className="spinner" /> 回测运行中...</> : '▶ 开始回测'}
            </button>
          </div>

          {polling && (
            <div className="card" style={{ textAlign: 'center', padding: 40 }}>
              <div className="spinner" style={{ width: 32, height: 32, margin: '0 auto 16px' }} />
              <p style={{ color: 'var(--text2)' }}>正在扫描 {stockPools.find(p => p.name === selectedPool)?.count || 5062} 只股票（{selectedPool}），请稍候...</p>
            </div>
          )}
        </div>
      )}

      {/* ===================== 回测结果 ===================== */}
      {tab === 'result' && result && (
        <div>
          <StatsSummary summary={result.summary} taskId={result.task_id} />
          {result.summary.backtest_mode === 'portfolio' && (
            <>
              <EquityCurveChart data={result.equity_curve} />
              <ReturnBreakdown annual={result.annual_returns} monthly={result.monthly_returns} />
            </>
          )}
          <TradeList trades={result.trades} mode={result.summary.backtest_mode} onViewKline={openKline} />
        </div>
      )}

      {/* ===================== K线标注 ===================== */}
      {tab === 'kline' && klineCode && (
        <KLineChart code={klineCode} kType={result?.summary.k_type || 'daily'} startDate={result?.summary.start_date} endDate={result?.summary.end_date} taskId={result?.task_id} />
      )}

      {/* ===================== 策略编辑 Modal ===================== */}
      {showCreate && (
        <div className="modal-overlay" onClick={() => setShowCreate(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2 style={{ marginBottom: 16 }}>{editName ? '编辑策略' : '新建策略'}</h2>
            <div className="form-group">
              <label>策略名称</label>
              <input value={editName} onChange={e => setEditName(e.target.value)} placeholder="输入策略名称" />
            </div>
            <div className="form-group config-editor">
              <label>策略配置 (JSON)</label>
              <textarea
                value={editYaml}
                onChange={e => setEditYaml(e.target.value)}
                placeholder='{"name": "my_strategy", "k_type": "daily", ...}'
              />
            </div>
            <div style={{ fontSize: 12, color: 'var(--text2)', marginBottom: 12 }}>
              支持字段：name, k_type, backtest_mode, buy_groups, sell_groups, add_groups(加仓), reduce_groups(减仓), reduce_pct, position_pct, max_positions, stop_loss_pct, take_profit_pct, trailing_stop_pct, stock_pool, buy_price_type(open/high/low/close/avg/typical/vwap), sell_price_type, buy_execution(same_day/next_day), sell_execution
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button className="btn btn-outline" onClick={() => setShowCreate(false)}>取消</button>
              <button className="btn btn-primary" onClick={saveStrategy}>保存</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/* ===================== 统计摘要 ===================== */
function StatsSummary({ summary, taskId }: { summary: BacktestSummary; taskId: string }) {
  return (
    <div className="card">
      <div className="flex-between mb-16">
        <h2>
          {summary.config_name}
          <span style={{ fontSize: 12, marginLeft: 8, padding: '2px 8px', background: 'var(--surface2)', borderRadius: 4, color: 'var(--text2)' }}>
            {summary.k_type}
          </span>
        </h2>
        <span style={{ fontSize: 12, color: 'var(--text2)' }}>{summary.start_date} ~ {summary.end_date}</span>
      </div>

      <div className="stats-grid">
        <StatItem label="胜率" value={summary.win_rate.toFixed(1) + '%'} />
        <StatItem label="盈亏比" value={summary.profit_loss_ratio.toFixed(2)} />
        <StatItem label="期望值" value={summary.expected_value.toFixed(2) + '%'} cls={summary.expected_value >= 0 ? 'positive' : 'negative'} />
        <StatItem label="交易次数" value={String(summary.total_trades)} />
        <StatItem label="盈利次数" value={String(summary.win_trades)} cls="positive" />
        <StatItem label="亏损次数" value={String(summary.lose_trades)} cls="negative" />
        <StatItem label="平均盈利" value={summary.avg_profit_pct.toFixed(2) + '%'} cls={summary.avg_profit_pct >= 0 ? 'positive' : 'negative'} />
        <StatItem label="最大盈利" value={summary.max_profit_pct.toFixed(2) + '%'} cls="positive" />
        <StatItem label="最大亏损" value={summary.max_loss_pct.toFixed(2) + '%'} cls="negative" />
        {summary.backtest_mode === 'portfolio' && (
          <>
            <StatItem label="总收益率" value={summary.total_return_pct.toFixed(2) + '%'} cls={summary.total_return_pct >= 0 ? 'positive' : 'negative'} />
            <StatItem label="年化收益率" value={summary.annual_return_pct.toFixed(2) + '%'} cls={summary.annual_return_pct >= 0 ? 'positive' : 'negative'} />
            <StatItem label="最大回撤" value={summary.max_drawdown_pct.toFixed(2) + '%'} cls="negative" />
            <StatItem label="Sharpe" value={summary.sharpe_ratio.toFixed(2)} />
            <StatItem label="初始资金" value={'¥' + summary.initial_capital.toLocaleString()} />
            <StatItem label="最终资金" value={'¥' + summary.final_capital.toLocaleString()} />
          </>
        )}
        {summary.avg_hold_days > 0 && <StatItem label="平均持仓(天)" value={summary.avg_hold_days.toFixed(0)} />}
      </div>
    </div>
  )
}

function StatItem({ label, value, cls }: { label: string; value: string; cls?: string }) {
  return (
    <div className="stat-item">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${cls || ''}`}>{value}</div>
    </div>
  )
}

/* ===================== 权益曲线 ===================== */
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

function EquityCurveChart({ data }: { data: EquityPoint[] }) {
  if (!data || data.length < 2) return null

  const formatted = data.map(d => ({
    ...d,
    equity: Number(d.equity.toFixed(0)),
  }))

  return (
    <div className="card">
      <h2>权益曲线</h2>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={formatted}>
            <defs>
              <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#33333a" />
            <XAxis dataKey="date" stroke="#a1a1aa" fontSize={11} tickFormatter={v => v.slice(5)} />
            <YAxis stroke="#a1a1aa" fontSize={11} tickFormatter={v => (v / 10000).toFixed(0) + 'w'} />
            <Tooltip
              contentStyle={{ background: '#1a1a1f', border: '1px solid #33333a', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#a1a1aa' }}
              formatter={(v: number) => ['¥' + v.toLocaleString(), '权益']}
            />
            <Area type="monotone" dataKey="equity" stroke="#3b82f6" fill="url(#eqGrad)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/* ===================== 收益分解 ===================== */
function ReturnBreakdown({ annual, monthly }: { annual: any[]; monthly: any[] }) {
  if ((!annual || annual.length === 0) && (!monthly || monthly.length === 0)) return null

  return (
    <div className="card">
      <h2>收益分解</h2>
      <div style={{ display: 'flex', gap: 24 }}>
        {monthly && monthly.length > 0 && (
          <div style={{ flex: 1 }}>
            <h3>月度收益</h3>
            <div className="table-wrap">
              <table>
                <thead><tr><th>月份</th><th>收益率</th></tr></thead>
                <tbody>
                  {monthly.map(m => (
                    <tr key={m.period}>
                      <td>{m.period}</td>
                      <td className={m.return_pct >= 0 ? 'positive' : 'negative'}>{m.return_pct.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {annual && annual.length > 0 && (
          <div style={{ flex: 1 }}>
            <h3>年度收益</h3>
            <div className="table-wrap">
              <table>
                <thead><tr><th>年度</th><th>收益率</th></tr></thead>
                <tbody>
                  {annual.map(a => (
                    <tr key={a.period}>
                      <td>{a.period}</td>
                      <td className={a.return_pct >= 0 ? 'positive' : 'negative'}>{a.return_pct.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/* ===================== 交易列表 ===================== */
function TradeList({ trades, mode, onViewKline }: { trades: Trade[]; mode: string; onViewKline: (code: string) => void }) {
  if (!trades || trades.length === 0) {
    return <div className="card"><div className="empty">暂无交易记录</div></div>
  }
  const isSignal = mode !== 'portfolio'

  // 按action统计
  const actionCounts = trades.reduce((acc, t) => {
    const a = t.action || (t.profit_pct >= 0 ? 'clear' : 'clear')
    acc[a] = (acc[a] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  return (
    <div className="card">
      <div className="flex-between mb-16">
        <h2>交易明细 ({trades.length} 笔)</h2>
        <div style={{ display: 'flex', gap: 8, fontSize: 12 }}>
          {Object.entries(actionCounts).map(([action, count]) => {
            const info = ACTION_LABELS[action] || { label: action, color: '#888' }
            return (
              <span key={action} style={{ padding: '2px 8px', borderRadius: 4, background: info.color + '22', color: info.color }}>
                {info.label} {count}
              </span>
            )
          })}
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>交易ID</th>
              <th>代码</th>
              <th>操作</th>
              <th>买入日</th>
              <th>买入价</th>
              <th>卖出日</th>
              <th>卖出价</th>
              {!isSignal && <th>股数</th>}
              <th>收益率</th>
              {!isSignal && <th>盈亏金额</th>}
              <th>持仓天数</th>
              <th>卖出原因</th>
              <th>K线</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => {
              const actionInfo = ACTION_LABELS[t.action] || ACTION_LABELS.clear
              return (
                <tr key={i}>
                  <td style={{ fontFamily: 'monospace', fontSize: 11, color: '#888' }}>{t.trade_id ? t.trade_id.slice(0, 6) : '-'}</td>
                  <td style={{ fontWeight: 600 }}>{t.code}</td>
                  <td>
                    <span style={{ padding: '1px 6px', borderRadius: 3, background: actionInfo.color + '22', color: actionInfo.color, fontSize: 12 }}>
                      {actionInfo.label}
                    </span>
                  </td>
                  <td>{t.buy_date}</td>
                  <td>{t.buy_price.toFixed(2)}</td>
                  <td>{t.sell_date}</td>
                  <td>{t.sell_price.toFixed(2)}</td>
                  {!isSignal && <td>{t.shares}</td>}
                  <td className={t.profit_pct >= 0 ? 'positive' : 'negative'}>{t.profit_pct >= 0 ? '+' : ''}{t.profit_pct.toFixed(2)}%</td>
                  {!isSignal && <td className={t.profit_amount >= 0 ? 'positive' : 'negative'}>¥{t.profit_amount.toFixed(0)}</td>}
                  <td>{t.hold_days}天</td>
                  <td style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={t.sell_reason}>
                    {t.sell_reason}
                  </td>
                  <td>
                    <button className="btn btn-outline" style={{ padding: '2px 8px', fontSize: 12 }} onClick={() => onViewKline(t.code)}>
                      📊
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ===================== K线图 + 交易标注 ===================== */
function KLineChart({ code, kType, startDate, endDate, taskId }: { code: string; kType: string; startDate?: string; endDate?: string; taskId?: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const [kline, setKline] = useState<KLinePoint[]>([])
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [loading, setLoading] = useState(true)
  const [dateRange, setDateRange] = useState({ start: startDate || '', end: endDate || '' })

  const fetchKline = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ k_type: kType })
      if (dateRange.start) params.set('start_date', dateRange.start)
      if (dateRange.end) params.set('end_date', dateRange.end)
      if (taskId) params.set('task_id', taskId)
      const res = await fetch(`${API}/kline/${code}?${params}`)
      const data = await res.json()
      setKline(data.kline || [])
      setAnnotations(data.annotations || [])
    } catch (e) {
      console.error('Failed to fetch kline:', e)
    }
    setLoading(false)
  }, [code, kType, dateRange])

  useEffect(() => { fetchKline() }, [fetchKline])

  // 渲染K线图
  useEffect(() => {
    if (!containerRef.current || kline.length === 0) return

    // 清除旧图
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const container = containerRef.current

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: '#1a1a1f' },
        textColor: '#a1a1aa',
      },
      grid: {
        vertLines: { color: '#2a2a30' },
        horzLines: { color: '#2a2a30' },
      },
      width: container.clientWidth,
      height: 500,
      timeScale: {
        timeVisible: false,
        borderColor: '#33333a',
      },
      rightPriceScale: {
        borderColor: '#33333a',
      },
    })

    chartRef.current = chart

    // K线
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#ef4444',
      downColor: '#22c55e',
      borderUpColor: '#ef4444',
      borderDownColor: '#22c55e',
      wickUpColor: '#ef4444',
      wickDownColor: '#22c55e',
    })

    const klineData: CandlestickData[] = kline.map(k => ({
      time: k.date as Time,
      open: k.open,
      high: k.high,
      low: k.low,
      close: k.close,
    }))
    candlestickSeries.setData(klineData)

    // 成交量
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    })
    volumeSeries.setData(kline.map(k => ({
      time: k.date as Time,
      value: k.volume,
      color: k.close >= k.open ? '#ef444466' : '#22c55e66',
    })))

    // 交易标注（markers）
    // 按 trade_id 分配颜色，同一笔交易的所有标注用相同颜色
    const TRADE_COLORS = [
      '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899',
      '#14b8a6', '#f97316', '#06b6d4', '#84cc16', '#e879f9',
      '#fb923c', '#38bdf8', '#a3e635', '#c084fc', '#f472b6',
    ]
    const tradeIdColorMap = new Map<string, string>()
    let colorIdx = 0
    const getTradeColor = (tid: string) => {
      if (!tradeIdColorMap.has(tid)) {
        tradeIdColorMap.set(tid, TRADE_COLORS[colorIdx % TRADE_COLORS.length])
        colorIdx++
      }
      return tradeIdColorMap.get(tid)!
    }

    const markers = annotations.map(ann => {
      const info = ACTION_LABELS[ann.type] || ACTION_LABELS.clear
      let text = info.label
      // 显示 trade_id 短标识（取前6位）
      if (ann.trade_id) {
        text = `[${ann.trade_id.slice(0, 6)}] ${text}`
      }
      if (ann.profit_pct !== undefined) {
        text += ` ${ann.profit_pct >= 0 ? '+' : ''}${ann.profit_pct.toFixed(1)}%`
      }
      if (ann.reason && ann.type !== 'buy' && ann.type !== 'add') {
        text += ` (${ann.reason})`
      }
      // 同一笔交易用同一颜色
      const markerColor = ann.trade_id ? getTradeColor(ann.trade_id) : info.color
      return {
        time: ann.date as Time,
        position: (ann.type === 'buy' || ann.type === 'add') ? 'belowBar' as const : 'aboveBar' as const,
        color: markerColor,
        shape: (ann.type === 'buy' || ann.type === 'add') ? 'arrowUp' as const : 'arrowDown' as const,
        text,
      }
    })

    // 按时间排序 markers
    markers.sort((a, b) => (a.time as string).localeCompare(b.time as string))

    // 去重：同一天同类型只保留一个（避免重复标注）
    const seen = new Set<string>()
    const uniqueMarkers = markers.filter(m => {
      const key = `${m.time}-${m.shape}-${m.text}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })

    candlestickSeries.setMarkers(uniqueMarkers)

    // 自适应宽度
    const handleResize = () => {
      if (chartRef.current && container) {
        chartRef.current.applyOptions({ width: container.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [kline, annotations])

  // 按action类型统计标注
  const annCounts = annotations.reduce((acc, a) => {
    acc[a.type] = (acc[a.type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  return (
    <div className="card">
      <div className="flex-between mb-16">
        <h2>
          {code} K线标注
          <span style={{ fontSize: 12, marginLeft: 8, padding: '2px 8px', background: 'var(--surface2)', borderRadius: 4, color: 'var(--text2)' }}>
            {kType}
          </span>
        </h2>
        <div style={{ display: 'flex', gap: 8 }}>
          {Object.entries(annCounts).map(([type, count]) => {
            const info = ACTION_LABELS[type] || { label: type, color: '#888' }
            return (
              <span key={type} style={{ padding: '2px 8px', borderRadius: 4, background: info.color + '22', color: info.color, fontSize: 12 }}>
                {info.label} {count}
              </span>
            )
          })}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <div className="form-group" style={{ flex: 1 }}>
          <label>开始日期</label>
          <input type="date" value={dateRange.start} onChange={e => setDateRange(d => ({ ...d, start: e.target.value }))} />
        </div>
        <div className="form-group" style={{ flex: 1 }}>
          <label>结束日期</label>
          <input type="date" value={dateRange.end} onChange={e => setDateRange(d => ({ ...d, end: e.target.value }))} />
        </div>
        <div className="form-group">
          <label>&nbsp;</label>
          <button className="btn btn-outline" onClick={fetchKline}>刷新</button>
        </div>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}>
          <div className="spinner" style={{ width: 24, height: 24, margin: '0 auto 12px' }} />
          <p style={{ color: 'var(--text2)' }}>加载K线数据...</p>
        </div>
      ) : kline.length === 0 ? (
        <div className="empty">无K线数据</div>
      ) : (
        <div ref={containerRef} style={{ borderRadius: 8, overflow: 'hidden' }} />
      )}

      {/* 图例 */}
      <div style={{ display: 'flex', gap: 16, marginTop: 12, fontSize: 12, color: 'var(--text2)' }}>
        {Object.entries(ACTION_LABELS).map(([key, info]) => (
          <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: info.color, display: 'inline-block' }} />
            {info.label}
          </div>
        ))}
      </div>
    </div>
  )
}
