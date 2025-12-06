"""
Execution Agent

This module implements an execution agent that:
- Receives trading decisions from DecisionMakingAgent
- Executes trades (simulated or real)
- Records and confirms trade execution
- Manages order tracking
"""

import logging
import json
from typing import Optional, Dict, List
from datetime import datetime
import os
import pandas as pd
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Execution agent that executes trading decisions.
    
    Receives decisions from DecisionMakingAgent and executes them,
    recording all trades for tracking and reporting.
    """
    
    def __init__(
        self,
        execution_mode: str = "simulated",  # "simulated" or "real"
        trade_log_path: str = "./trades_log.json",
        enable_slippage: bool = True,
        slippage_factor: float = 0.001,  # 0.1% slippage
        commission_rate: float = 0.001,  # 0.1% commission
        history_size: int = 1000
    ):
        """
        Initialize execution agent.
        
        Args:
            execution_mode: "simulated" for paper trading, "real" for live trading
            trade_log_path: Path to save trade log
            enable_slippage: Whether to simulate slippage
            slippage_factor: Slippage factor (0.001 = 0.1%)
            commission_rate: Commission rate per trade
            history_size: Size of execution history
        """
        self.execution_mode = execution_mode
        self.trade_log_path = trade_log_path
        self.enable_slippage = enable_slippage
        self.slippage_factor = slippage_factor
        self.commission_rate = commission_rate
        
        # Execution history
        self.history_size = history_size
        self.execution_history: deque = deque(maxlen=history_size)
        
        # Pending orders
        self.pending_orders: List[Dict] = []
        
        # Trade log
        self.trades: List[Dict] = []
        self._load_trade_log()
        
        logger.info(f"Initialized ExecutionAgent (mode: {execution_mode})")
    
    def receive_decision(self, decision: Dict) -> Dict:
        """
        Receives trading decision from DecisionMakingAgent and executes it.
        
        Args:
            decision: Dictionary with decision:
                {
                    "action": "BUY" | "SELL" | "HOLD",
                    "ticker": str,
                    "quantity": int,
                    "price": float,
                    "confidence": float,
                    "timestamp": str,
                    ...
                }
        
        Returns:
            Dictionary with execution result:
                {
                    "status": "executed" | "rejected" | "pending",
                    "order_id": str,
                    "ticker": str,
                    "action": str,
                    "quantity": int,
                    "requested_price": float,
                    "executed_price": float,
                    "commission": float,
                    "slippage": float,
                    "timestamp": str,
                    "message": str
                }
        """
        try:
            action = decision.get("action", "HOLD")
            ticker = decision.get("ticker", "UNKNOWN")
            
            logger.info(f"Received decision: {action} {decision.get('quantity', 0)} {ticker}")
            
            # Handle HOLD
            if action == "HOLD":
                return self._create_hold_response(decision)
            
            # Validate decision
            validation_result = self._validate_decision(decision)
            if not validation_result["valid"]:
                return self._create_rejected_response(decision, validation_result["reason"])
            
            # Execute trade
            if self.execution_mode == "simulated":
                execution_result = self._execute_simulated_trade(decision)
            else:
                execution_result = self._execute_real_trade(decision)
            
            # Record execution
            self._record_execution(execution_result)
            
            # Save to log
            self._save_trade_log()
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return self._create_error_response(decision, str(e))
    
    def _validate_decision(self, decision: Dict) -> Dict:
        """Validate trading decision before execution."""
        action = decision.get("action")
        quantity = decision.get("quantity", 0)
        price = decision.get("price", 0.0)
        ticker = decision.get("ticker", "")
        
        # Check required fields
        if not ticker:
            return {"valid": False, "reason": "Missing ticker"}
        
        if action not in ["BUY", "SELL"]:
            return {"valid": False, "reason": f"Invalid action: {action}"}
        
        if quantity <= 0:
            return {"valid": False, "reason": f"Invalid quantity: {quantity}"}
        
        if price <= 0:
            return {"valid": False, "reason": f"Invalid price: {price}"}
        
        # Additional validation for real trading
        if self.execution_mode == "real":
            # Add real broker validation here
            pass
        
        return {"valid": True}
    
    def _execute_simulated_trade(self, decision: Dict) -> Dict:
        """Execute trade in simulated mode."""
        action = decision.get("action")
        ticker = decision.get("ticker")
        quantity = decision.get("quantity", 0)
        requested_price = decision.get("price", 0.0)
        
        # Simulate slippage
        if self.enable_slippage:
            slippage = requested_price * self.slippage_factor
            if action == "BUY":
                # Buy at slightly higher price
                executed_price = requested_price + slippage
            else:
                # Sell at slightly lower price
                executed_price = requested_price - slippage
        else:
            executed_price = requested_price
            slippage = 0.0
        
        # Calculate commission
        trade_value = quantity * executed_price
        commission = trade_value * self.commission_rate
        
        # Generate order ID
        order_id = self._generate_order_id()
        
        # Create execution result
        execution_result = {
            "status": "executed",
            "order_id": order_id,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "requested_price": requested_price,
            "executed_price": executed_price,
            "slippage": slippage,
            "commission": commission,
            "total_cost": trade_value + commission if action == "BUY" else trade_value - commission,
            "timestamp": datetime.now().isoformat() + "Z",
            "execution_mode": "simulated",
            "message": f"Trade executed successfully (simulated)"
        }
        
        logger.info(f"Simulated trade executed: {action} {quantity} {ticker} @ {executed_price:.2f}")
        
        return execution_result
    
    def _execute_real_trade(self, decision: Dict) -> Dict:
        """Execute trade in real mode (requires broker API integration)."""
        # This would integrate with a real broker API
        # For now, return error
        logger.warning("Real trading not implemented. Use simulated mode.")
        return self._create_rejected_response(
            decision,
            "Real trading not implemented. Please use simulated mode."
        )
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"ORD_{timestamp}"
    
    def _record_execution(self, execution_result: Dict):
        """Record execution in history and trades log."""
        # Add to history
        self.execution_history.append(execution_result.copy())
        
        # Add to trades log
        trade_record = {
            "order_id": execution_result["order_id"],
            "ticker": execution_result["ticker"],
            "action": execution_result["action"],
            "quantity": execution_result["quantity"],
            "executed_price": execution_result["executed_price"],
            "commission": execution_result["commission"],
            "timestamp": execution_result["timestamp"],
            "status": execution_result["status"]
        }
        self.trades.append(trade_record)
    
    def _create_hold_response(self, decision: Dict) -> Dict:
        """Create response for HOLD decision."""
        return {
            "status": "hold",
            "order_id": None,
            "ticker": decision.get("ticker", "UNKNOWN"),
            "action": "HOLD",
            "quantity": 0,
            "requested_price": 0.0,
            "executed_price": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
            "timestamp": datetime.now().isoformat() + "Z",
            "message": "HOLD decision - no trade executed"
        }
    
    def _create_rejected_response(self, decision: Dict, reason: str) -> Dict:
        """Create response for rejected decision."""
        return {
            "status": "rejected",
            "order_id": None,
            "ticker": decision.get("ticker", "UNKNOWN"),
            "action": decision.get("action", "UNKNOWN"),
            "quantity": decision.get("quantity", 0),
            "requested_price": decision.get("price", 0.0),
            "executed_price": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
            "timestamp": datetime.now().isoformat() + "Z",
            "message": f"Trade rejected: {reason}"
        }
    
    def _create_error_response(self, decision: Dict, error: str) -> Dict:
        """Create response for error."""
        return {
            "status": "error",
            "order_id": None,
            "ticker": decision.get("ticker", "UNKNOWN"),
            "action": decision.get("action", "UNKNOWN"),
            "quantity": 0,
            "requested_price": 0.0,
            "executed_price": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
            "timestamp": datetime.now().isoformat() + "Z",
            "message": f"Execution error: {error}"
        }
    
    def _load_trade_log(self):
        """Load trade log from file."""
        if os.path.exists(self.trade_log_path):
            try:
                with open(self.trade_log_path, 'r') as f:
                    self.trades = json.load(f)
                logger.info(f"Loaded {len(self.trades)} trades from log")
            except Exception as e:
                logger.warning(f"Error loading trade log: {e}")
                self.trades = []
        else:
            self.trades = []
    
    def _save_trade_log(self):
        """Save trade log to file."""
        try:
            os.makedirs(os.path.dirname(self.trade_log_path) if os.path.dirname(self.trade_log_path) else ".", exist_ok=True)
            with open(self.trade_log_path, 'w') as f:
                json.dump(self.trades, f, indent=2)
            logger.debug(f"Trade log saved ({len(self.trades)} trades)")
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")
    
    def get_execution_history(self, n: Optional[int] = None) -> List[Dict]:
        """Get execution history."""
        history = list(self.execution_history)
        if n:
            return history[-n:]
        return history
    
    def get_trade_statistics(self) -> Dict:
        """Get trade statistics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "total_volume": 0,
                "total_commission": 0.0
            }
        
        buy_trades = [t for t in self.trades if t.get("action") == "BUY"]
        sell_trades = [t for t in self.trades if t.get("action") == "SELL"]
        
        total_volume = sum(t.get("quantity", 0) * t.get("executed_price", 0.0) for t in self.trades)
        total_commission = sum(t.get("commission", 0.0) for t in self.trades)
        
        return {
            "total_trades": len(self.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_volume": total_volume,
            "total_commission": total_commission,
            "avg_commission": total_commission / len(self.trades) if self.trades else 0.0
        }
    
    def get_trades_by_ticker(self, ticker: str) -> List[Dict]:
        """Get all trades for a specific ticker."""
        return [t for t in self.trades if t.get("ticker") == ticker]
    
    def confirm_execution(self, order_id: str) -> Dict:
        """Confirm trade execution (for real trading)."""
        # Find order in history
        for execution in self.execution_history:
            if execution.get("order_id") == order_id:
                return {
                    "confirmed": True,
                    "order_id": order_id,
                    "status": execution.get("status"),
                    "message": "Execution confirmed"
                }
        
        return {
            "confirmed": False,
            "order_id": order_id,
            "message": "Order not found"
        }


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = ExecutionAgent(execution_mode="simulated")
    
    # Example decision (from DecisionMakingAgent)
    example_decision = {
        "action": "BUY",
        "ticker": "AAPL",
        "quantity": 10,
        "price": 150.0,
        "confidence": 0.75,
        "timestamp": datetime.now().isoformat() + "Z"
    }
    
    # Execute trade
    result = agent.receive_decision(example_decision)
    print("\n" + "="*80)
    print("EXECUTION RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("="*80)
    
    # Get statistics
    stats = agent.get_trade_statistics()
    print("\nTRADE STATISTICS:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

