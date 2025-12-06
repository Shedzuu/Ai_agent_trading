"""
Complete Multi-Agent Trading System Example

Demonstrates the full workflow:
1. Market Monitoring Agent - fetches and analyzes market data
2. Decision Making Agent - makes trading decisions using AI
3. Execution Agent - executes trades
4. Coordinator - manages the workflow
"""

import json
import time
from agents.market_monitor import MarketMonitoringAgent
from agents.decision_maker import DecisionMakingAgent
from agents.execution_agent import ExecutionAgent
from agents.coordinator import AgentCoordinator


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def example_1_manual_workflow():
    """Example 1: Manual step-by-step workflow."""
    print_section("Example 1: Manual Step-by-Step Workflow")
    
    ticker = "AAPL"
    
    # Step 1: Market Monitoring Agent
    print("Step 1: Market Monitoring Agent")
    print("-" * 80)
    market_agent = MarketMonitoringAgent(
        ticker=ticker,
        interval="1h",
        period="1mo"
    )
    
    data, analysis = market_agent.get_processed_data(analyze=True)
    print(f"✓ Fetched {len(data)} data points")
    print(f"✓ Market trend: {analysis['trend']}")
    print(f"✓ Trend strength: {analysis['strength']:.2f}")
    
    # Send to decision agent
    market_message = market_agent.send_to_decision_agent(transport="direct")
    print(f"✓ Market data prepared for Decision Agent")
    
    # Step 2: Decision Making Agent
    print("\nStep 2: Decision Making Agent")
    print("-" * 80)
    decision_agent = DecisionMakingAgent(
        model_type="random_forest",
        risk_tolerance="medium",
        enable_ai=True
    )
    
    decision = decision_agent.receive_market_data(market_message)
    print(f"✓ Decision: {decision['action']}")
    print(f"✓ Confidence: {decision['confidence']:.2f}")
    print(f"✓ Reasoning: {decision['reasoning']}")
    if decision['action'] != 'HOLD':
        print(f"✓ Quantity: {decision['quantity']}")
        print(f"✓ Price: ${decision['price']:.2f}")
    
    # Step 3: Execution Agent
    print("\nStep 3: Execution Agent")
    print("-" * 80)
    execution_agent = ExecutionAgent(execution_mode="simulated")
    
    execution_result = execution_agent.receive_decision(decision)
    print(f"✓ Status: {execution_result['status']}")
    print(f"✓ Message: {execution_result['message']}")
    
    if execution_result['status'] == 'executed':
        print(f"✓ Order ID: {execution_result['order_id']}")
        print(f"✓ Executed Price: ${execution_result['executed_price']:.2f}")
        print(f"✓ Commission: ${execution_result['commission']:.2f}")
        
        # Update portfolio
        decision_agent.update_portfolio(
            ticker,
            execution_result['action'],
            execution_result['quantity'],
            execution_result['executed_price']
        )
    
    # Show portfolio
    print("\nPortfolio Status:")
    print("-" * 80)
    portfolio = decision_agent.get_portfolio_status()
    print(json.dumps(portfolio, indent=2, ensure_ascii=False))
    
    # Show trade statistics
    print("\nTrade Statistics:")
    print("-" * 80)
    stats = execution_agent.get_trade_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def example_2_coordinator_workflow():
    """Example 2: Using Coordinator for automated workflow."""
    print_section("Example 2: Coordinator Automated Workflow")
    
    ticker = "TSLA"
    
    # Create coordinator
    print(f"Creating coordinator for {ticker}...")
    coordinator = AgentCoordinator(
        ticker=ticker,
        auto_start=False  # Don't start continuous monitoring
    )
    print("✓ Coordinator created")
    
    # Run single cycle
    print(f"\nRunning trading cycle for {ticker}...")
    result = coordinator.run_single_cycle()
    
    print("\nWorkflow Result:")
    print("-" * 80)
    print(f"Ticker: {result['ticker']}")
    print(f"Market Price: ${result['market_data']['price']:.2f}")
    print(f"Market Trend: {result['market_data']['trend']}")
    print(f"Decision: {result['decision']['action']}")
    print(f"Confidence: {result['decision']['confidence']:.2f}")
    print(f"Execution Status: {result['execution']['status']}")
    
    if result['execution']['status'] == 'executed':
        print(f"Order ID: {result['execution']['order_id']}")
        print(f"Executed Price: ${result['execution']['executed_price']:.2f}")
    
    # Show system status
    print("\nSystem Status:")
    print("-" * 80)
    status = coordinator.get_system_status()
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))


def example_3_multiple_tickers():
    """Example 3: Process multiple tickers."""
    print_section("Example 3: Processing Multiple Tickers")
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        print("-" * 80)
        
        try:
            coordinator = AgentCoordinator(ticker=ticker, auto_start=False)
            result = coordinator.run_single_cycle()
            
            print(f"✓ {ticker}: {result['decision']['action']} "
                  f"(confidence: {result['decision']['confidence']:.2f})")
            
            if result['execution']['status'] == 'executed':
                print(f"  → Executed: {result['execution']['quantity']} shares @ "
                      f"${result['execution']['executed_price']:.2f}")
        except Exception as e:
            print(f"✗ Error processing {ticker}: {e}")


def example_4_decision_history():
    """Example 4: View decision and execution history."""
    print_section("Example 4: Decision and Execution History")
    
    ticker = "AAPL"
    
    # Create agents
    market_agent = MarketMonitoringAgent(ticker=ticker, interval="1d", period="1mo")
    decision_agent = DecisionMakingAgent()
    execution_agent = ExecutionAgent()
    
    # Run multiple cycles
    print(f"Running 5 cycles for {ticker}...")
    for i in range(5):
        print(f"\nCycle {i+1}:")
        # Get processed data first
        market_agent.get_processed_data(analyze=True)
        market_message = market_agent.send_to_decision_agent(transport="direct")
        decision = decision_agent.receive_market_data(market_message)
        execution = execution_agent.receive_decision(decision)
        
        print(f"  Decision: {decision['action']} (confidence: {decision['confidence']:.2f})")
        print(f"  Execution: {execution['status']}")
        
        time.sleep(0.5)  # Small delay
    
    # Show history
    print("\nDecision History (last 5):")
    print("-" * 80)
    decisions = decision_agent.get_decision_history(n=5)
    for d in decisions:
        print(f"  {d['timestamp']}: {d['action']} {d['ticker']} "
              f"(confidence: {d['confidence']:.2f})")
    
    print("\nExecution History (last 5):")
    print("-" * 80)
    executions = execution_agent.get_execution_history(n=5)
    for e in executions:
        if e['status'] == 'executed':
            print(f"  {e['timestamp']}: {e['action']} {e['quantity']} {e['ticker']} "
                  f"@ ${e['executed_price']:.2f}")
        else:
            print(f"  {e['timestamp']}: {e['status']} - {e['message']}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("  MULTI-AGENT TRADING SYSTEM - COMPLETE WORKFLOW EXAMPLES")
    print("="*80)
    
    try:
        # Example 1: Manual workflow
        example_1_manual_workflow()
        
        time.sleep(2)
        
        # Example 2: Coordinator workflow
        example_2_coordinator_workflow()
        
        time.sleep(2)
        
        # Example 3: Multiple tickers
        example_3_multiple_tickers()
        
        time.sleep(2)
        
        # Example 4: History
        example_4_decision_history()
        
        print_section("All Examples Completed Successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

