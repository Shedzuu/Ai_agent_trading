"""
MarketMonitoringAgent Usage Examples

Demonstrates all main agent capabilities:
- Basic usage
- Market conditions analysis
- Anomaly detection
- Continuous monitoring with alerts
- Batch processing
"""

from agents.market_monitor import MarketMonitoringAgent
import json
import time
import threading

# ============================================================================
# Example 1: Basic usage with analysis
# ============================================================================
print("=" * 80)
print("Example 1: Basic usage with analysis")
print("=" * 80)

agent_aapl = MarketMonitoringAgent(
    ticker="AAPL",
    interval="1h",
    period="1mo",
    enable_cache=True,
    indicators=["sma", "rsi", "macd", "bb"]
)

try:
    # Get data with analysis
    data, analysis = agent_aapl.get_processed_data(analyze=True)
    
    print(f"\nRecords obtained: {len(data)}")
    print("\nFirst 5 records:")
    print(data.head())
    
    print("\nMarket conditions analysis:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    # Generate insights
    insights = agent_aapl.generate_insights(data, analysis)
    print("\nInsights:")
    print(json.dumps(insights, indent=2, ensure_ascii=False))
    
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 2: Anomaly detection
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Anomaly detection")
print("=" * 80)

agent_btc = MarketMonitoringAgent(
    ticker="BTC-USD",
    interval="1h",
    period="1mo"
)

try:
    data_btc = agent_btc.get_processed_data()
    anomalies = agent_btc.detect_anomalies(data_btc, window=10, z_thresh=2.5)
    
    print(f"\nAnomalies found: {len(anomalies)}")
    if anomalies:
        print("\nFirst 5 anomalies:")
        for anomaly in anomalies[:5]:
            print(json.dumps(anomaly, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 3: Alerts and monitoring
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Alerts and monitoring")
print("=" * 80)

agent_tsla = MarketMonitoringAgent(
    ticker="TSLA",
    interval="1d",
    period="3mo"
)

try:
    data_tsla = agent_tsla.get_processed_data()
    
    # Define alert callback
    def alert_callback(alert_data):
        print(f"\n🚨 ALERT for {alert_data['ticker']}:")
        print(json.dumps(alert_data, indent=2, ensure_ascii=False))
    
    agent_tsla.on_alert(alert_callback)
    
    # Check alerts with thresholds
    thresholds = {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "volatility": 0.02
    }
    
    alert = agent_tsla.should_alert(thresholds)
    if alert:
        alert_callback(alert)
    else:
        print("\nNo alerts detected")
    
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 4: Sending data to Decision Agent
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Sending data to Decision Agent")
print("=" * 80)

try:
    # Form standardized message
    message = agent_aapl.send_to_decision_agent(transport="direct")
    
    print("\nStandardized message for Decision Agent:")
    print(json.dumps(message, indent=2, ensure_ascii=False))
    
    # Handle request from another agent
    request = {"type": "get_latest", "n": 5}
    response = agent_aapl.handle_request(request)
    print("\nRequest response:")
    print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
    
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 5: Batch processing
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Batch processing (multiple tickers)")
print("=" * 80)

try:
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"\nProcessing {len(tickers)} tickers sequentially...")
    results_sequential = MarketMonitoringAgent.process_batch(
        tickers=tickers,
        interval="1d",
        period="1mo",
        parallel=False
    )
    
    print(f"\nResults:")
    for ticker, df in results_sequential.items():
        print(f"  {ticker}: {len(df)} records")
    
    print(f"\nProcessing {len(tickers)} tickers in parallel...")
    results_parallel = MarketMonitoringAgent.process_batch(
        tickers=tickers,
        interval="1d",
        period="1mo",
        parallel=True,
        max_workers=3
    )
    
    print(f"\nResults:")
    for ticker, df in results_parallel.items():
        print(f"  {ticker}: {len(df)} records")
    
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 6: Continuous monitoring (demonstration)
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Continuous monitoring (demonstration)")
print("=" * 80)

print("""
To start continuous monitoring use:

def alert_callback(alert_data):
    print(f"Alert: {alert_data}")

agent = MarketMonitoringAgent("AAPL", interval="1h", period="1mo")
agent.on_alert(alert_callback)

stop_event = threading.Event()
thresholds = {"rsi_overbought": 70, "rsi_oversold": 30}

# Start in separate thread
monitor_thread = threading.Thread(
    target=agent.monitor_continuously,
    args=(300, stop_event, thresholds)  # 300 seconds = 5 minutes
)
monitor_thread.start()

# To stop: stop_event.set()
""")

print("\n" + "=" * 80)
print("All examples completed!")
print("=" * 80)
