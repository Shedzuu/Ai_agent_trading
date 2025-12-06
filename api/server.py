"""
FastAPI Backend Server

REST API for multi-agent trading system.
Provides endpoints for:
- Market data
- Trading decisions
- Trade execution
- System status
"""

import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uvicorn

from agents.market_monitor import MarketMonitoringAgent
from agents.decision_maker import DecisionMakingAgent
from agents.execution_agent import ExecutionAgent
from agents.coordinator import AgentCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Trading System API",
    description="REST API for coordinating trading agents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinators (in production, use proper state management)
coordinators: Dict[str, AgentCoordinator] = {}


# Pydantic models for request/response
class MarketDataRequest(BaseModel):
    ticker: str
    interval: str = "1h"
    period: str = "1mo"


class DecisionRequest(BaseModel):
    market_data: Dict


class ExecutionRequest(BaseModel):
    decision: Dict


class StartMonitoringRequest(BaseModel):
    ticker: str
    interval_seconds: int = 300


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent Trading System API",
        "version": "1.0.0",
        "endpoints": {
            "market": "/api/market",
            "decision": "/api/decision",
            "execution": "/api/execution",
            "coordinator": "/api/coordinator",
            "status": "/api/status"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Market Monitoring Endpoints

@app.post("/api/market/data")
async def get_market_data(request: MarketDataRequest):
    """Get market data for a ticker."""
    try:
        agent = MarketMonitoringAgent(
            ticker=request.ticker,
            interval=request.interval,
            period=request.period
        )
        
        data, analysis = agent.get_processed_data(analyze=True)
        market_message = agent.send_to_decision_agent(transport="direct")
        
        return {
            "success": True,
            "ticker": request.ticker,
            "data_points": len(data),
            "market_data": market_message,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/data/{ticker}")
async def get_market_data_simple(ticker: str, interval: str = "1h", period: str = "1mo"):
    """Get market data (simple GET endpoint)."""
    try:
        agent = MarketMonitoringAgent(ticker=ticker, interval=interval, period=period)
        data, analysis = agent.get_processed_data(analyze=True)
        market_message = agent.send_to_decision_agent(transport="direct")
        
        return {
            "success": True,
            "ticker": ticker,
            "data_points": len(data),
            "market_data": market_message,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Decision Making Endpoints

@app.post("/api/decision/make")
async def make_decision(request: DecisionRequest):
    """Make trading decision based on market data."""
    try:
        agent = DecisionMakingAgent(
            model_type="random_forest",
            risk_tolerance="medium"
        )
        
        decision = agent.receive_market_data(request.market_data)
        
        return {
            "success": True,
            "decision": decision,
            "portfolio": agent.get_portfolio_status()
        }
    except Exception as e:
        logger.error(f"Error making decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Execution Endpoints

@app.post("/api/execution/execute")
async def execute_trade(request: ExecutionRequest):
    """Execute a trading decision."""
    try:
        agent = ExecutionAgent(execution_mode="simulated")
        result = agent.receive_decision(request.decision)
        
        return {
            "success": True,
            "execution": result,
            "statistics": agent.get_trade_statistics()
        }
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/execution/history")
async def get_execution_history(limit: int = 50):
    """Get execution history."""
    try:
        agent = ExecutionAgent()
        history = agent.get_execution_history(n=limit)
        stats = agent.get_trade_statistics()
        
        return {
            "success": True,
            "history": history,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting execution history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Coordinator Endpoints

@app.post("/api/coordinator/start")
async def start_coordinator(request: StartMonitoringRequest):
    """Start a coordinator for a ticker."""
    try:
        if request.ticker in coordinators:
            return {
                "success": False,
                "message": f"Coordinator for {request.ticker} already exists"
            }
        
        coordinator = AgentCoordinator(
            ticker=request.ticker,
            auto_start=True
        )
        coordinators[request.ticker] = coordinator
        
        return {
            "success": True,
            "ticker": request.ticker,
            "message": "Coordinator started"
        }
    except Exception as e:
        logger.error(f"Error starting coordinator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/coordinator/run-cycle/{ticker}")
async def run_coordinator_cycle(ticker: str):
    """Run a single cycle for a coordinator."""
    try:
        if ticker not in coordinators:
            raise HTTPException(status_code=404, detail=f"Coordinator for {ticker} not found")
        
        coordinator = coordinators[ticker]
        result = coordinator.run_single_cycle()
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error running cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/coordinator/status/{ticker}")
async def get_coordinator_status(ticker: str):
    """Get status of a coordinator."""
    try:
        if ticker not in coordinators:
            raise HTTPException(status_code=404, detail=f"Coordinator for {ticker} not found")
        
        coordinator = coordinators[ticker]
        status = coordinator.get_system_status()
        
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/coordinator/stop/{ticker}")
async def stop_coordinator(ticker: str):
    """Stop a coordinator."""
    try:
        if ticker not in coordinators:
            raise HTTPException(status_code=404, detail=f"Coordinator for {ticker} not found")
        
        coordinator = coordinators[ticker]
        coordinator.stop_monitoring()
        del coordinators[ticker]
        
        return {
            "success": True,
            "message": f"Coordinator for {ticker} stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping coordinator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/coordinator/list")
async def list_coordinators():
    """List all active coordinators."""
    return {
        "success": True,
        "coordinators": list(coordinators.keys()),
        "count": len(coordinators)
    }


# System Status Endpoint

@app.get("/api/status")
async def get_system_status():
    """Get overall system status."""
    return {
        "success": True,
        "system": {
            "status": "operational",
            "active_coordinators": len(coordinators),
            "timestamp": datetime.now().isoformat()
        },
        "coordinators": {
            ticker: coordinator.get_system_status()
            for ticker, coordinator in coordinators.items()
        }
    }


# Complete Workflow Endpoint

@app.post("/api/workflow/run")
async def run_complete_workflow(request: MarketDataRequest):
    """Run complete trading workflow: Market → Decision → Execution."""
    try:
        # Step 1: Market Monitoring
        market_agent = MarketMonitoringAgent(
            ticker=request.ticker,
            interval=request.interval,
            period=request.period
        )
        market_message = market_agent.send_to_decision_agent(transport="direct")
        
        # Step 2: Decision Making
        decision_agent = DecisionMakingAgent(
            model_type="random_forest",
            risk_tolerance="medium"
        )
        decision = decision_agent.receive_market_data(market_message)
        
        # Step 3: Execution
        execution_agent = ExecutionAgent(execution_mode="simulated")
        execution_result = execution_agent.receive_decision(decision)
        
        # Update portfolio if executed
        if execution_result.get("status") == "executed":
            action = execution_result.get("action")
            quantity = execution_result.get("quantity", 0)
            price = execution_result.get("executed_price", 0.0)
            decision_agent.update_portfolio(request.ticker, action, quantity, price)
        
        return {
            "success": True,
            "workflow": {
                "ticker": request.ticker,
                "timestamp": datetime.now().isoformat(),
                "market_data": market_message,
                "decision": decision,
                "execution": execution_result,
                "portfolio": decision_agent.get_portfolio_status()
            }
        }
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

