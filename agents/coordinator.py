"""
Agent Coordinator

This module implements a coordinator that manages communication and workflow
between all agents in the trading system.
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime
import threading
import time

from .market_monitor import MarketMonitoringAgent
from .decision_maker import DecisionMakingAgent
from .execution_agent import ExecutionAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates communication and workflow between agents.
    
    Manages the flow:
    MarketMonitoringAgent → DecisionMakingAgent → ExecutionAgent
    """
    
    def __init__(
        self,
        ticker: str,
        market_agent: Optional[MarketMonitoringAgent] = None,
        decision_agent: Optional[DecisionMakingAgent] = None,
        execution_agent: Optional[ExecutionAgent] = None,
        auto_start: bool = False
    ):
        """
        Initialize agent coordinator.
        
        Args:
            ticker: Stock ticker to monitor
            market_agent: MarketMonitoringAgent instance (creates new if None)
            decision_agent: DecisionMakingAgent instance (creates new if None)
            execution_agent: ExecutionAgent instance (creates new if None)
            auto_start: Whether to start monitoring automatically
        """
        self.ticker = ticker
        
        # Initialize agents
        self.market_agent = market_agent or MarketMonitoringAgent(
            ticker=ticker,
            interval="1h",
            period="1mo"
        )
        
        self.decision_agent = decision_agent or DecisionMakingAgent(
            model_type="random_forest",
            risk_tolerance="medium"
        )
        
        self.execution_agent = execution_agent or ExecutionAgent(
            execution_mode="simulated"
        )
        
        # State
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # History
        self.workflow_history: List[Dict] = []
        
        logger.info(f"Initialized AgentCoordinator for {ticker}")
        
        if auto_start:
            self.start_monitoring()
    
    def run_single_cycle(self) -> Dict:
        """
        Run a single cycle of the trading workflow.
        
        Returns:
            Dictionary with complete workflow result
        """
        try:
            logger.info(f"Running trading cycle for {self.ticker}")
            
            # Step 1: Market Monitoring Agent - Get market data
            logger.info("Step 1: Fetching market data...")
            market_data = self.market_agent.get_processed_data(analyze=True)
            
            if isinstance(market_data, tuple):
                data_df, analysis = market_data
            else:
                analysis = self.market_agent.analyze_market_conditions(market_data)
            
            # Format market data for decision agent
            market_message = self.market_agent.send_to_decision_agent(transport="direct")
            
            # Step 2: Decision Making Agent - Make decision
            logger.info("Step 2: Making trading decision...")
            decision = self.decision_agent.receive_market_data(market_message)
            
            # Step 3: Execution Agent - Execute trade
            logger.info("Step 3: Executing trade...")
            execution_result = self.execution_agent.receive_decision(decision)
            
            # Update portfolio if trade executed
            if execution_result.get("status") == "executed":
                action = execution_result.get("action")
                quantity = execution_result.get("quantity", 0)
                price = execution_result.get("executed_price", 0.0)
                self.decision_agent.update_portfolio(
                    self.ticker,
                    action,
                    quantity,
                    price
                )
            
            # Create workflow result
            workflow_result = {
                "timestamp": datetime.now().isoformat() + "Z",
                "ticker": self.ticker,
                "market_data": {
                    "price": market_message.get("ohlcv", {}).get("close", 0.0),
                    "trend": market_message.get("analysis", {}).get("trend", "unknown"),
                    "indicators": market_message.get("indicators", {})
                },
                "decision": decision,
                "execution": execution_result,
                "portfolio": self.decision_agent.get_portfolio_status()
            }
            
            # Store in history
            self.workflow_history.append(workflow_result)
            
            logger.info(f"Cycle completed. Decision: {decision.get('action')}, Status: {execution_result.get('status')}")
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {
                "timestamp": datetime.now().isoformat() + "Z",
                "ticker": self.ticker,
                "error": str(e),
                "status": "failed"
            }
    
    def start_monitoring(self, interval_seconds: int = 300):
        """
        Start continuous monitoring and trading.
        
        Args:
            interval_seconds: Interval between cycles in seconds
        """
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        def monitor_loop():
            logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
            while not self.stop_event.is_set():
                try:
                    self.run_single_cycle()
                    self.stop_event.wait(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    self.stop_event.wait(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Monitoring stopped")
    
    def get_workflow_history(self, n: Optional[int] = None) -> List[Dict]:
        """Get workflow history."""
        history = list(self.workflow_history)
        if n:
            return history[-n:]
        return history
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "ticker": self.ticker,
            "is_running": self.is_running,
            "portfolio": self.decision_agent.get_portfolio_status(),
            "trade_stats": self.execution_agent.get_trade_statistics(),
            "recent_decisions": self.decision_agent.get_decision_history(n=5),
            "recent_executions": self.execution_agent.get_execution_history(n=5)
        }


# Example usage
if __name__ == "__main__":
    # Create coordinator
    coordinator = AgentCoordinator(
        ticker="AAPL",
        auto_start=False
    )
    
    # Run single cycle
    result = coordinator.run_single_cycle()
    
    print("\n" + "="*80)
    print("WORKFLOW RESULT:")
    print("="*80)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("="*80)

