"""
Decision-Making Agent

This module implements a decision-making agent that:
- Receives market data from MarketMonitoringAgent
- Uses AI model to make trading decisions (BUY/SELL/HOLD)
- Applies risk management rules
- Sends decisions to ExecutionAgent
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json
from collections import deque
import pickle
import os

# AI/ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Using rule-based fallback.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DecisionMakingAgent:
    """
    Decision-making agent that uses AI to make trading decisions.
    
    Receives market data from MarketMonitoringAgent and makes decisions
    based on technical indicators, market analysis, and risk management rules.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        risk_tolerance: str = "medium",  # "low", "medium", "high"
        max_position_size: float = 0.1,  # Max 10% of portfolio per trade
        min_confidence: float = 0.6,  # Minimum confidence for action
        enable_ai: bool = True,
        model_path: Optional[str] = None,
        history_size: int = 1000,
        use_historical_training: bool = True,  # Use real historical data for training
        training_ticker: Optional[str] = None,  # Ticker for historical training
        training_period: str = "1y"  # Period for historical training data
    ):
        """
        Initialize decision-making agent.
        
        Args:
            model_type: Type of AI model ("random_forest", "gradient_boosting", "rule_based")
            risk_tolerance: Risk tolerance level ("low", "medium", "high")
            max_position_size: Maximum position size as fraction of portfolio
            min_confidence: Minimum confidence threshold for taking action
            enable_ai: Whether to use AI model (if False, uses rule-based)
            model_path: Path to saved model (if None, trains new model)
            history_size: Size of decision history
            use_historical_training: If True, train on real historical data (default: True)
            training_ticker: Ticker for historical training data (default: "SPY")
            training_period: Period for historical training ("1y", "6mo", "3mo", "1mo")
        """
        self.model_type = model_type
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.enable_ai = enable_ai and SKLEARN_AVAILABLE
        self.use_historical_training = use_historical_training
        self.training_ticker = training_ticker
        self.training_period = training_period
        
        # AI Model
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False
        
        # Decision history
        self.history_size = history_size
        self.decision_history: deque = deque(maxlen=history_size)
        
        # Portfolio state (simulated)
        self.portfolio = {
            "cash": 10000.0,  # Starting cash
            "positions": {},  # {ticker: {"quantity": int, "avg_price": float}}
            "total_value": 10000.0
        }
        
        # Risk management parameters
        self.risk_params = self._get_risk_params()
        
        # Initialize model
        if self.enable_ai:
            if model_path and os.path.exists(model_path):
                self._load_model(model_path)
            else:
                logger.info("AI model will be trained on first decision")
        else:
            logger.info("Using rule-based decision making (AI not available)")
    
    def _get_risk_params(self) -> Dict:
        """Get risk management parameters based on risk tolerance."""
        params = {
            "low": {
                "max_loss_per_trade": 0.01,  # 1% max loss
                "stop_loss": 0.02,  # 2% stop loss
                "take_profit": 0.05,  # 5% take profit
                "max_drawdown": 0.1  # 10% max drawdown
            },
            "medium": {
                "max_loss_per_trade": 0.02,  # 2% max loss
                "stop_loss": 0.03,  # 3% stop loss
                "take_profit": 0.08,  # 8% take profit
                "max_drawdown": 0.15  # 15% max drawdown
            },
            "high": {
                "max_loss_per_trade": 0.03,  # 3% max loss
                "stop_loss": 0.05,  # 5% stop loss
                "take_profit": 0.12,  # 12% take profit
                "max_drawdown": 0.25  # 25% max drawdown
            }
        }
        return params.get(self.risk_tolerance, params["medium"])
    
    def receive_market_data(self, market_data: Dict) -> Dict:
        """
        Receives market data from MarketMonitoringAgent and makes decision.
        
        Args:
            market_data: Dictionary with market data in standardized format:
                {
                    "timestamp": str,
                    "ticker": str,
                    "ohlcv": {...},
                    "indicators": {...},
                    "analysis": {...},
                    "meta": {...}
                }
        
        Returns:
            Dictionary with trading decision:
                {
                    "action": "BUY" | "SELL" | "HOLD",
                    "ticker": str,
                    "confidence": float (0.0-1.0),
                    "reasoning": str,
                    "quantity": int (if action is BUY/SELL),
                    "price": float,
                    "timestamp": str,
                    "risk_score": float
                }
        """
        try:
            logger.info(f"Received market data for {market_data.get('ticker', 'UNKNOWN')}")
            
            # Extract features for AI model
            features = self._extract_features(market_data)
            
            # Make decision using AI or rules
            if self.enable_ai and self.is_trained:
                decision = self._make_ai_decision(features, market_data)
            else:
                decision = self._make_rule_based_decision(features, market_data)
            
            # Apply risk management
            decision = self._apply_risk_management(decision, market_data)
            
            # Store in history
            self.decision_history.append({
                "timestamp": decision["timestamp"],
                "ticker": decision["ticker"],
                "action": decision["action"],
                "confidence": decision["confidence"],
                "price": decision["price"]
            })
            
            # Train model if needed (online learning)
            if self.enable_ai and not self.is_trained:
                self._train_initial_model()
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return self._create_hold_decision(market_data, f"Error: {str(e)}")
    
    def _extract_features(self, market_data: Dict) -> np.ndarray:
        """Extract features from market data for AI model."""
        indicators = market_data.get("indicators", {})
        analysis = market_data.get("analysis", {})
        ohlcv = market_data.get("ohlcv", {})
        
        # Feature vector
        features = []
        
        # Price features
        features.append(ohlcv.get("close", 0.0))
        features.append(ohlcv.get("volume", 0.0))
        features.append(indicators.get("price_change", 0.0))
        
        # Technical indicators
        features.append(indicators.get("sma10", 0.0))
        features.append(indicators.get("sma20", 0.0))
        features.append(indicators.get("rsi14", 50.0))
        features.append(indicators.get("macd", 0.0))
        features.append(indicators.get("macd_hist", 0.0))
        features.append(indicators.get("volatility", 0.0))
        
        # Analysis features
        trend = analysis.get("trend", "sideways")
        trend_encoded = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}.get(trend, 0.0)
        features.append(trend_encoded)
        features.append(analysis.get("strength", 0.5))
        
        # RSI state
        rsi_state = analysis.get("signals", {}).get("rsi_state", "neutral")
        rsi_encoded = {"overbought": 1.0, "oversold": -1.0, "neutral": 0.0}.get(rsi_state, 0.0)
        features.append(rsi_encoded)
        
        # SMA crossover
        sma_cross = analysis.get("signals", {}).get("sma_cross", 0)
        features.append(float(sma_cross))
        
        return np.array(features).reshape(1, -1)
    
    def _make_ai_decision(self, features: np.ndarray, market_data: Dict) -> Dict:
        """Make decision using trained AI model."""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Map prediction to action
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            action = action_map.get(prediction, "HOLD")
            
            # Get confidence (probability of predicted class)
            confidence = float(max(probabilities))
            
            # Generate reasoning
            reasoning = self._generate_reasoning(market_data, action, confidence)
            
            ticker = market_data.get("ticker", "UNKNOWN")
            price = market_data.get("ohlcv", {}).get("close", 0.0)
            
            return {
                "action": action,
                "ticker": ticker,
                "confidence": confidence,
                "reasoning": reasoning,
                "quantity": self._calculate_quantity(action, price, confidence),
                "price": price,
                "timestamp": datetime.now().isoformat() + "Z",
                "risk_score": self._calculate_risk_score(market_data),
                "model_type": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error in AI decision: {e}")
            return self._make_rule_based_decision(features, market_data)
    
    def _make_rule_based_decision(self, features: np.ndarray, market_data: Dict) -> Dict:
        """Make decision using rule-based logic (fallback or when AI disabled)."""
        indicators = market_data.get("indicators", {})
        analysis = market_data.get("analysis", {})
        ohlcv = market_data.get("ohlcv", {})
        
        ticker = market_data.get("ticker", "UNKNOWN")
        price = ohlcv.get("close", 0.0)
        
        # Extract key signals
        trend = analysis.get("trend", "sideways")
        rsi = indicators.get("rsi14", 50.0)
        rsi_state = analysis.get("signals", {}).get("rsi_state", "neutral")
        macd_hist = indicators.get("macd_hist", 0.0)
        sma_cross = analysis.get("signals", {}).get("sma_cross", 0)
        strength = analysis.get("strength", 0.5)
        
        # Decision logic
        buy_signals = 0
        sell_signals = 0
        
        # Trend signals
        if trend == "bull":
            buy_signals += 2
        elif trend == "bear":
            sell_signals += 2
        
        # RSI signals
        if rsi_state == "oversold" and rsi < 30:
            buy_signals += 2
        elif rsi_state == "overbought" and rsi > 70:
            sell_signals += 2
        
        # MACD signals
        if macd_hist > 0:
            buy_signals += 1
        elif macd_hist < 0:
            sell_signals += 1
        
        # SMA crossover
        if sma_cross:
            if trend == "bull":
                buy_signals += 1
            elif trend == "bear":
                sell_signals += 1
        
        # Strength multiplier
        strength_mult = strength
        
        # Calculate confidence
        total_signals = buy_signals + sell_signals
        if total_signals == 0:
            action = "HOLD"
            confidence = 0.5
        elif buy_signals > sell_signals:
            action = "BUY"
            confidence = min(0.9, 0.5 + (buy_signals / 10.0) * strength_mult)
        elif sell_signals > buy_signals:
            action = "SELL"
            confidence = min(0.9, 0.5 + (sell_signals / 10.0) * strength_mult)
        else:
            action = "HOLD"
            confidence = 0.5
        
        reasoning = self._generate_reasoning(market_data, action, confidence)
        
        return {
            "action": action,
            "ticker": ticker,
            "confidence": confidence,
            "reasoning": reasoning,
            "quantity": self._calculate_quantity(action, price, confidence),
            "price": price,
            "timestamp": datetime.now().isoformat() + "Z",
            "risk_score": self._calculate_risk_score(market_data),
            "model_type": "rule_based"
        }
    
    def _generate_reasoning(self, market_data: Dict, action: str, confidence: float) -> str:
        """Generate human-readable reasoning for decision."""
        indicators = market_data.get("indicators", {})
        analysis = market_data.get("analysis", {})
        
        trend = analysis.get("trend", "sideways")
        rsi = indicators.get("rsi14", 50.0)
        rsi_state = analysis.get("signals", {}).get("rsi_state", "neutral")
        strength = analysis.get("strength", 0.5)
        
        reasons = []
        
        if action == "BUY":
            reasons.append(f"Trend: {trend.upper()}")
            if rsi_state == "oversold":
                reasons.append(f"RSI oversold ({rsi:.1f})")
            if strength > 0.6:
                reasons.append(f"Strong trend (strength: {strength:.2f})")
        elif action == "SELL":
            reasons.append(f"Trend: {trend.upper()}")
            if rsi_state == "overbought":
                reasons.append(f"RSI overbought ({rsi:.1f})")
            if strength > 0.6:
                reasons.append(f"Strong trend (strength: {strength:.2f})")
        else:
            reasons.append("Mixed signals or low confidence")
            reasons.append(f"Trend: {trend}, RSI: {rsi:.1f}")
        
        reasoning = f"{action} decision (confidence: {confidence:.2f}). " + ". ".join(reasons)
        return reasoning
    
    def _calculate_quantity(self, action: str, price: float, confidence: float) -> int:
        """Calculate quantity to trade based on confidence and risk management."""
        if action == "HOLD":
            return 0
        
        # Calculate position size based on confidence and risk tolerance
        portfolio_value = self.portfolio.get("total_value", 10000.0)
        max_position_value = portfolio_value * self.max_position_size
        
        # Adjust by confidence
        confidence_multiplier = confidence
        position_value = max_position_value * confidence_multiplier
        
        # Calculate quantity
        if price > 0:
            quantity = int(position_value / price)
            return max(1, quantity)  # At least 1 share
        
        return 0
    
    def _calculate_risk_score(self, market_data: Dict) -> float:
        """Calculate risk score (0.0-1.0, higher = riskier)."""
        indicators = market_data.get("indicators", {})
        volatility = indicators.get("volatility", 0.0)
        price = market_data.get("ohlcv", {}).get("close", 0.0)
        
        # Normalize volatility
        if price > 0:
            volatility_pct = (volatility / price) * 100
        else:
            volatility_pct = 0.0
        
        # Risk score based on volatility (0-1 scale)
        risk_score = min(1.0, volatility_pct / 5.0)  # 5% volatility = max risk
        
        return risk_score
    
    def _apply_risk_management(self, decision: Dict, market_data: Dict) -> Dict:
        """Apply risk management rules to decision."""
        action = decision.get("action", "HOLD")
        confidence = decision.get("confidence", 0.0)
        risk_score = decision.get("risk_score", 0.5)
        
        # Check minimum confidence
        if confidence < self.min_confidence and action != "HOLD":
            logger.info(f"Decision rejected: confidence {confidence:.2f} < min {self.min_confidence}")
            return self._create_hold_decision(market_data, "Low confidence")
        
        # Check risk score
        max_risk = self.risk_params.get("max_drawdown", 0.15)
        if risk_score > max_risk and action != "HOLD":
            logger.info(f"Decision rejected: risk score {risk_score:.2f} > max {max_risk}")
            return self._create_hold_decision(market_data, "Risk too high")
        
        # Adjust quantity based on risk
        if action in ["BUY", "SELL"]:
            risk_multiplier = 1.0 - (risk_score * 0.5)  # Reduce position by up to 50% based on risk
            decision["quantity"] = int(decision["quantity"] * risk_multiplier)
            decision["quantity"] = max(1, decision["quantity"])
        
        # Add stop loss and take profit
        if action in ["BUY", "SELL"]:
            price = decision.get("price", 0.0)
            decision["stop_loss"] = price * (1 - self.risk_params["stop_loss"])
            decision["take_profit"] = price * (1 + self.risk_params["take_profit"])
        
        return decision
    
    def _create_hold_decision(self, market_data: Dict, reason: str = "") -> Dict:
        """Create a HOLD decision."""
        ticker = market_data.get("ticker", "UNKNOWN")
        price = market_data.get("ohlcv", {}).get("close", 0.0)
        
        return {
            "action": "HOLD",
            "ticker": ticker,
            "confidence": 0.5,
            "reasoning": f"HOLD: {reason}" if reason else "HOLD: No clear signal",
            "quantity": 0,
            "price": price,
            "timestamp": datetime.now().isoformat() + "Z",
            "risk_score": self._calculate_risk_score(market_data),
            "model_type": "rule_based" if not self.enable_ai else self.model_type
        }
    
    def _train_initial_model(self):
        """Train initial AI model using historical or synthetic data."""
        if not self.enable_ai:
            return
        
        logger.info("Training initial AI model...")
        
        # Try to use historical data if enabled
        if self.use_historical_training:
            try:
                logger.info("Attempting to train on real historical data...")
                X, y = self._prepare_historical_training_data()
                if X is not None and len(X) > 100:  # Need at least 100 samples
                    logger.info(f"Using {len(X)} historical samples for training")
                else:
                    logger.warning("Not enough historical data, falling back to synthetic data")
                    X, y = self._prepare_synthetic_training_data()
            except Exception as e:
                logger.warning(f"Error preparing historical data: {e}. Using synthetic data.")
                X, y = self._prepare_synthetic_training_data()
        else:
            logger.info("Using synthetic training data")
            X, y = self._prepare_synthetic_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model trained. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        self.is_trained = True
        
        # Save model if path provided
        if self.model_path:
            self._save_model(self.model_path)
    
    def _prepare_historical_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from real historical market data.
        
        Returns:
            Tuple of (features, labels) arrays
        """
        try:
            from agents.market_monitor import MarketMonitoringAgent
            
            # Use provided ticker or default to SPY (broad market index)
            ticker = self.training_ticker or "SPY"
            
            logger.info(f"Fetching historical data for {ticker} (period: {self.training_period})")
            
            # Get historical data
            market_agent = MarketMonitoringAgent(
                ticker=ticker,
                interval="1d",  # Daily data for training
                period=self.training_period,
                enable_cache=True
            )
            
            # Get processed data with indicators
            data = market_agent.get_processed_data(analyze=False)
            
            if data.empty or len(data) < 50:
                logger.warning(f"Insufficient historical data: {len(data)} records")
                return None, None
            
            # Prepare features and labels
            X = []
            y = []
            
            # For each day, extract features and determine label based on future price movement
            for i in range(len(data) - 1):
                current_row = data.iloc[i]
                next_row = data.iloc[i + 1]
                
                # Extract features (same as _extract_features but from DataFrame row)
                features = []
                
                # Price features
                features.append(float(current_row.get('close', 0.0)))
                features.append(float(current_row.get('volume', 0.0)))
                features.append(float(current_row.get('price_change', 0.0)))
                
                # Technical indicators
                features.append(float(current_row.get('sma10', 0.0)))
                features.append(float(current_row.get('sma20', 0.0)))
                features.append(float(current_row.get('rsi14', 50.0)))
                features.append(float(current_row.get('macd', 0.0)))
                features.append(float(current_row.get('macd_hist', 0.0)))
                features.append(float(current_row.get('volatility', 0.0)))
                
                # Analysis features (need to compute)
                sma10 = current_row.get('sma10', 0.0)
                sma20 = current_row.get('sma20', 0.0)
                if sma10 > sma20:
                    trend_encoded = 1.0  # bull
                elif sma10 < sma20:
                    trend_encoded = -1.0  # bear
                else:
                    trend_encoded = 0.0  # sideways
                features.append(trend_encoded)
                
                # Strength (based on RSI distance from 50)
                rsi = current_row.get('rsi14', 50.0)
                strength = abs(rsi - 50) / 50
                features.append(float(strength))
                
                # RSI state
                if rsi > 70:
                    rsi_encoded = 1.0  # overbought
                elif rsi < 30:
                    rsi_encoded = -1.0  # oversold
                else:
                    rsi_encoded = 0.0  # neutral
                features.append(rsi_encoded)
                
                # SMA crossover (simplified)
                sma_cross = 0.0
                if i > 0:
                    prev_row = data.iloc[i - 1]
                    prev_sma10 = prev_row.get('sma10', 0.0)
                    prev_sma20 = prev_row.get('sma20', 0.0)
                    if (prev_sma10 <= prev_sma20 and sma10 > sma20) or \
                       (prev_sma10 >= prev_sma20 and sma10 < sma20):
                        sma_cross = 1.0
                features.append(sma_cross)
                
                # Determine label based on future price movement
                current_price = current_row.get('close', 0.0)
                next_price = next_row.get('close', 0.0)
                
                if current_price > 0:
                    price_change_pct = ((next_price - current_price) / current_price) * 100
                    
                    # Label: 0=SELL, 1=HOLD, 2=BUY
                    # Based on future return and current indicators
                    if price_change_pct > 1.0 and rsi < 60:  # Price goes up, not overbought
                        label = 2  # BUY
                    elif price_change_pct < -1.0 and rsi > 40:  # Price goes down, not oversold
                        label = 0  # SELL
                    else:
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD if no price data
                
                X.append(features)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared {len(X)} training samples from historical data")
            logger.info(f"Label distribution: BUY={np.sum(y==2)}, HOLD={np.sum(y==1)}, SELL={np.sum(y==0)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing historical training data: {e}")
            return None, None
    
    def _prepare_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare synthetic training data based on rules.
        
        Returns:
            Tuple of (features, labels) arrays
        """
        logger.info("Generating synthetic training data...")
        
        n_samples = 1000
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate random features
            features = np.random.rand(1, 14)[0]
            
            # Apply rule-based logic to generate labels
            trend = features[9]  # Trend feature
            rsi = features[5] * 100  # RSI (0-100)
            macd_hist = features[7]  # MACD histogram
            
            # Rule-based label
            if trend > 0.3 and rsi < 40 and macd_hist > 0:
                label = 2  # BUY
            elif trend < -0.3 and rsi > 60 and macd_hist < 0:
                label = 0  # SELL
            else:
                label = 1  # HOLD
            
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def _save_model(self, path: str):
        """Save trained model to disk."""
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                    "model_type": self.model_type
                }, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self, path: str):
        """Load trained model from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.model_type = data.get("model_type", self.model_type)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    def update_portfolio(self, ticker: str, action: str, quantity: int, price: float):
        """Update portfolio state after trade execution."""
        if action == "BUY":
            cost = quantity * price
            if cost <= self.portfolio["cash"]:
                self.portfolio["cash"] -= cost
                if ticker in self.portfolio["positions"]:
                    # Average price calculation
                    old_qty = self.portfolio["positions"][ticker]["quantity"]
                    old_price = self.portfolio["positions"][ticker]["avg_price"]
                    total_cost = (old_qty * old_price) + cost
                    total_qty = old_qty + quantity
                    self.portfolio["positions"][ticker] = {
                        "quantity": total_qty,
                        "avg_price": total_cost / total_qty
                    }
                else:
                    self.portfolio["positions"][ticker] = {
                        "quantity": quantity,
                        "avg_price": price
                    }
        elif action == "SELL":
            if ticker in self.portfolio["positions"]:
                qty = self.portfolio["positions"][ticker]["quantity"]
                if quantity <= qty:
                    revenue = quantity * price
                    self.portfolio["cash"] += revenue
                    if quantity == qty:
                        del self.portfolio["positions"][ticker]
                    else:
                        self.portfolio["positions"][ticker]["quantity"] -= quantity
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        return self.portfolio.copy()
    
    def get_decision_history(self, n: Optional[int] = None) -> List[Dict]:
        """Get decision history."""
        history = list(self.decision_history)
        if n:
            return history[-n:]
        return history


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = DecisionMakingAgent(
        model_type="random_forest",
        risk_tolerance="medium",
        enable_ai=True
    )
    
    # Example market data (from MarketMonitoringAgent)
    example_market_data = {
        "timestamp": datetime.now().isoformat() + "Z",
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
            "rsi14": 45.0,
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
                "rsi_state": "neutral"
            },
            "strength": 0.65
        },
        "meta": {
            "source": "yfinance",
            "fetched_at": datetime.now().isoformat() + "Z"
        }
    }
    
    # Make decision
    decision = agent.receive_market_data(example_market_data)
    print("\n" + "="*80)
    print("DECISION:")
    print("="*80)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print("="*80)

