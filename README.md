# AI Agent Monitoring PM

A market monitoring project using agent-based architecture.

## Installation

```bash
pip install -r requirements.txt
```

---

## 🤖 How This AI Agent Works

### What Makes It an AI Agent?

Unlike a simple script, `MarketMonitoringAgent` demonstrates **autonomous agent behavior**:

1. **Decision Making**: Analyzes market data and makes decisions about trends, signals, and alerts
2. **Autonomy**: Can run continuously in background, monitoring markets without human intervention
3. **Memory**: Maintains state history to compare current vs. previous market conditions
4. **Reactivity**: Responds to market events (price spikes, threshold breaches) with alerts
5. **Inter-agent Communication**: Sends structured data to Decision Agent and handles requests
6. **Adaptability**: Retries on failures, uses cache fallback, validates data automatically

### Core Workflow

```
1. Fetch Data → 2. Compute Indicators → 3. Analyze → 4. Generate Insights → 5. Send to Decision Agent
```

---

## 📊 What Data We Extract

### Raw Market Data (OHLCV)
- **Open, High, Low, Close**: Price data for each time period
- **Volume**: Trading volume
- **Timestamp**: When the data was recorded

### Technical Indicators Computed

**Trend Indicators:**
- `sma10`, `sma20` - Simple Moving Averages (10 and 20 periods)
- `macd`, `macd_signal`, `macd_hist` - MACD indicator

**Momentum Indicators:**
- `rsi14` - Relative Strength Index (0-100, overbought >70, oversold <30)
- `price_change` - Percentage price change

**Volatility Indicators:**
- `volatility` - Standard deviation of price over 10 periods
- `bb_upper`, `bb_lower`, `bb_mid` - Bollinger Bands

### Analysis Output

The agent generates **structured analysis**:

```python
{
  "trend": "bull" | "bear" | "sideways",
  "signals": {
    "sma_cross": 0 or 1,  # Whether SMA crossover occurred
    "rsi_state": "overbought" | "oversold" | "neutral"
  },
  "strength": 0.0-1.0  # Trend strength (higher = stronger)
}
```

---

## 🎯 What the Data Shows

### Market Trend
- **Bull**: Price is rising (SMA10 > SMA20)
- **Bear**: Price is falling (SMA10 < SMA20)
- **Sideways**: No clear direction

### Trading Signals
- **SMA Crossover**: When short-term MA crosses long-term MA (potential trend change)
- **RSI State**: 
  - Overbought (>70): Price may be too high, potential sell signal
  - Oversold (<30): Price may be too low, potential buy signal
  - Neutral (30-70): Normal market conditions

### Market Strength
- **0.0-0.3**: Weak trend
- **0.3-0.7**: Moderate trend
- **0.7-1.0**: Strong trend

### Anomalies Detected
- **Price spikes**: Unusual price movements (Z-score > threshold)
- **Volume spikes**: Unusual trading volume

---

## 🚀 How to Use

### Basic Usage

```python
from agents.market_monitor import MarketMonitoringAgent

# Create agent
agent = MarketMonitoringAgent(
    ticker="AAPL",      # Stock ticker
    interval="1h",      # Candle interval (1h, 30m, 1d, 5m, 15m)
    period="1mo"        # Data period (1mo, 3mo, 1y, 6mo, 1d)
)

# Get processed data
data = agent.get_processed_data()
print(data.head())
```

### With Analysis

```python
# Get data with analysis
data, analysis = agent.get_processed_data(analyze=True)

print(analysis)
# {
#   "trend": "bull",
#   "signals": {
#     "sma_cross": 0,
#     "rsi_state": "overbought"
#   },
#   "strength": 0.72
# }
```

### Anomaly Detection

```python
anomalies = agent.detect_anomalies(data, window=10, z_thresh=3.0)
# Returns list of anomalies with type, value, and z-score
```

### Continuous Monitoring

```python
import threading

def alert_callback(alert_data):
    print(f"Alert: {alert_data}")

agent.on_alert(alert_callback)

stop_event = threading.Event()
thresholds = {
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "volatility": 0.02
}

# Start in separate thread
monitor_thread = threading.Thread(
    target=agent.monitor_continuously,
    args=(300, stop_event, thresholds)  # 300 seconds = 5 minutes
)
monitor_thread.start()

# To stop: stop_event.set()
```

### Batch Processing

```python
# Process multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL"]

# Sequentially
results = MarketMonitoringAgent.process_batch(
    tickers=tickers,
    interval="1d",
    period="1mo",
    parallel=False
)

# In parallel
results = MarketMonitoringAgent.process_batch(
    tickers=tickers,
    interval="1d",
    period="1mo",
    parallel=True,
    max_workers=4
)
```

### Integration with Decision Agent

```python
# Send data in standardized format
message = agent.send_to_decision_agent(transport="direct")
# Returns JSON-serializable dict

# Handle requests from other agents
request = {"type": "get_latest", "n": 10}
response = agent.handle_request(request)
```

---

## 📤 Data Format for Decision Agent

The agent sends data to Decision Agent in this standardized JSON format:

```json
{
  "timestamp": "2025-12-05T12:00:00Z",
  "ticker": "AAPL",
  "ohlcv": {
    "open": 150.0,
    "high": 152.0,
    "low": 149.0,
    "close": 151.0,
    "volume": 1000000
  },
  "indicators": {
    "sma10": 150.5,
    "sma20": 149.8,
    "rsi14": 65.0,
    "macd": -0.5,
    "macd_signal": 0.2,
    "macd_hist": -0.7,
    "bb_upper": 155.0,
    "bb_lower": 145.0,
    "bb_mid": 150.0,
    "price_change": 0.5,
    "volatility": 1.2
  },
  "analysis": {
    "trend": "bull",
    "signals": {
      "sma_cross": 0,
      "rsi_state": "overbought"
    },
    "strength": 0.72
  },
  "meta": {
    "source": "yfinance",
    "fetched_at": "2025-12-05T12:00:00Z"
  }
}
```

**Decision Agent receives:**
- Raw OHLCV data
- All computed indicators
- Pre-analyzed market conditions (trend, signals, strength)
- Metadata for tracking

---

## 🏗️ Agent Architecture

### Current Implementation: Market Monitoring Agent (Agent 1)

```
MarketMonitoringAgent
    ↓
[Fetches Data] → [Computes Indicators] → [Analyzes] → [Sends to Decision Agent]
```

**Responsibilities:**
- ✅ Fetch market data from Yahoo Finance
- ✅ Compute technical indicators (SMA, RSI, MACD, Bollinger Bands)
- ✅ Analyze market conditions (trend, signals, strength)
- ✅ Detect anomalies
- ✅ Monitor continuously with alerts
- ✅ Send structured data to Decision Agent

---

## 🔮 Future Agents (Not Yet Implemented)

### Decision Agent (Agent 2) - Next in Chain

**Expected Responsibilities:**
- Receive data from MarketMonitoringAgent
- Make trading decisions (BUY/SELL/HOLD) based on:
  - Market analysis from Agent 1
  - Technical indicators
  - Risk management rules
  - Portfolio constraints
- Send decisions to ExecutionAgent

**How to Integrate:**
```python
# MarketMonitoringAgent sends data
message = agent.send_to_decision_agent(transport="direct")

# Decision Agent should:
# 1. Receive this message format
# 2. Parse analysis and indicators
# 3. Apply decision logic
# 4. Return decision: {"action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, ...}
```

**Expected Input Format:**
- Uses the JSON message format shown above
- Focuses on `analysis.trend`, `analysis.signals`, and `indicators` fields
- May request additional data via `handle_request()`

---

### Execution Agent (Agent 3) - Final in Chain

**Expected Responsibilities:**
- Receive decisions from DecisionAgent
- Execute trades (if applicable)
- Manage order placement
- Track execution results
- Send reports to ReportingAgent

**How to Integrate:**
```python
# Decision Agent sends decision
decision = {"action": "BUY", "ticker": "AAPL", "quantity": 10, ...}

# Execution Agent should:
# 1. Receive decision from DecisionAgent
# 2. Validate and execute trade
# 3. Report execution status
```

---

## 📋 Project Criteria (Agent 1 - Market Monitoring)

### ✅ Implemented Features

1. **Data Collection**
   - ✅ Fetches real market data via Yahoo Finance API
   - ✅ Supports multiple timeframes (1h, 30m, 1d, etc.)
   - ✅ Handles errors with retry mechanism
   - ✅ Caches data for performance

2. **Data Processing**
   - ✅ Computes technical indicators (SMA, RSI, MACD, Bollinger Bands)
   - ✅ Preprocesses data (removes NaN, normalizes)
   - ✅ Validates data quality

3. **Agent Capabilities**
   - ✅ Analyzes market conditions (trend, signals, strength)
   - ✅ Detects anomalies automatically
   - ✅ Generates insights and summaries
   - ✅ Monitors continuously in background
   - ✅ Sends alerts based on thresholds
   - ✅ Maintains state history

4. **Integration**
   - ✅ Standardized data format for Decision Agent
   - ✅ Request handling from other agents
   - ✅ Batch processing for multiple tickers

### 📝 Data Output Structure

**DataFrame Columns:**
- `timestamp` - Time when data was recorded
- `open`, `high`, `low`, `close` - OHLC prices
- `volume` - Trading volume
- `sma10`, `sma20` - Moving averages
- `rsi14` - RSI indicator (0-100)
- `macd`, `macd_signal`, `macd_hist` - MACD components
- `bb_upper`, `bb_lower`, `bb_mid` - Bollinger Bands
- `price_change` - Percentage change
- `volatility` - Price volatility

**Analysis Dictionary:**
- `trend`: Market direction (bull/bear/sideways)
- `signals`: Trading signals (SMA crossover, RSI state)
- `strength`: Trend strength (0.0-1.0)

---

## 🔧 Configuration

```python
agent = MarketMonitoringAgent(
    ticker="AAPL",
    interval="1h",
    period="1mo",
    enable_cache=True,           # Enable caching
    cache_path="./cache",        # Cache directory path
    history_size=100,            # State history size
    indicators=["sma", "rsi", "macd", "bb"],  # Indicators to compute
    max_retries=3,              # Retry attempts on errors
    backoff_factor=2.0          # Exponential backoff multiplier
)
```

---

## ⚠️ Limitations & Recommendations

- **Rate Limits**: Yahoo Finance API has request rate limitations
- **Recommended Interval**: For intraday data (1m, 5m) use `interval_seconds >= 60`
- **Memory**: Consider memory requirements when processing many tickers
- **Cache TTL**: Default 1 hour (3600 seconds)

---

## 📚 Examples

See `example_usage.py` for complete usage examples of all features.

---

## 📦 Dependencies

- `yfinance>=0.2.28` - Market data from Yahoo Finance
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Mathematical operations
- `pyarrow>=10.0.0` - Parquet caching (optional, falls back to CSV)

---

## 🔗 Agent Chain Architecture

```
┌─────────────────────┐
│ MarketMonitoringAgent│ ← Currently Implemented
│    (Agent 1)         │
│                      │
│ • Fetches data       │
│ • Computes indicators│
│ • Analyzes conditions│
│ • Detects anomalies  │
└──────────┬───────────┘
           │
           │ Sends: JSON message with OHLCV, indicators, analysis
           ↓
┌─────────────────────┐
│  DecisionAgent      │ ← To Be Implemented
│    (Agent 2)        │
│                      │
│ • Receives analysis  │
│ • Makes decisions    │
│ • Risk management    │
└──────────┬───────────┘
           │
           │ Sends: Trading decision (BUY/SELL/HOLD)
           ↓
┌─────────────────────┐
│ ExecutionAgent      │ ← To Be Implemented
│    (Agent 3)        │
│                      │
│ • Executes trades    │
│ • Manages orders     │
│ • Reports results    │
└─────────────────────┘
```

---

## 💡 Key Points for Next Agents

### For Decision Agent Developer:

1. **Input Format**: Use `send_to_decision_agent()` output - it's already standardized
2. **Key Fields to Use**:
   - `analysis.trend` - Market direction
   - `analysis.signals.rsi_state` - Overbought/oversold signals
   - `analysis.signals.sma_cross` - Trend change signals
   - `indicators.rsi14` - Current RSI value
   - `indicators.macd_hist` - MACD momentum
3. **Request Data**: Use `handle_request()` to get historical data if needed
4. **Response Format**: Return decision as JSON: `{"action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reasoning": "..."}`

### For Execution Agent Developer:

1. **Input**: Receive decision from DecisionAgent
2. **Integration**: May need to query MarketMonitoringAgent for latest prices
3. **Output**: Execution status and results

---

## 🎓 Understanding the Agent's Intelligence

**What makes it "intelligent":**
- **Pattern Recognition**: Identifies trends and signals from raw data
- **Anomaly Detection**: Finds unusual patterns using statistical methods
- **Decision Making**: Determines when to alert based on thresholds
- **Adaptive Behavior**: Retries on failures, uses cache when API fails
- **Context Awareness**: Maintains history to compare current vs. past states

**It's not just a script because:**
- It **interprets** data (not just processes)
- It **decides** when alerts are needed
- It **learns** from failures (retry logic)
- It **communicates** with other agents
- It **works autonomously** (continuous monitoring)
