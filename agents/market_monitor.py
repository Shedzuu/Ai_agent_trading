"""
Market Monitoring Agent

This module implements a market monitoring agent that:
- Fetches market data via Yahoo Finance API
- Computes technical indicators
- Analyzes market conditions
- Preprocesses data for transmission to other agents
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
import time
import json
import os
from typing import Optional, List, Dict, Tuple, Callable, Union
from collections import deque
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketMonitoringAgent:
    """
    Market monitoring agent.
    
    Fetches market data, computes technical indicators,
    analyzes market conditions and returns structured data
    for further processing.
    """
    
    def __init__(
        self,
        ticker: str,
        interval: str = "1h",
        period: str = "1mo",
        enable_cache: bool = True,
        cache_path: str = "./cache",
        history_size: int = 100,
        indicators: Optional[List[str]] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """
        Initialize market monitoring agent.
        
        Args:
            ticker: Instrument ticker (e.g., 'AAPL', 'TSLA', 'BTC-USD')
            interval: Candle time interval ('1h', '30m', '1d', '5m', '15m')
            period: Data period ('1mo', '3mo', '1y', '6mo', '1d')
            enable_cache: Enable data caching
            cache_path: Path to cache directory
            history_size: State history size (default 100)
            indicators: List of indicators to compute (None = all available)
            max_retries: Maximum number of retry attempts on errors
            backoff_factor: Exponential backoff multiplier
        """
        self.ticker = ticker.upper()
        self.interval = interval
        self.period = period
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
        # Caching
        self.enable_cache = enable_cache
        self.cache_path = cache_path
        self.cache_ttl = 3600  # 1 hour by default
        
        # State history
        self.history_size = history_size
        self.state_history: deque = deque(maxlen=history_size)
        
        # Indicators
        self.indicators = indicators if indicators else ["sma", "rsi", "macd", "bb"]
        
        # Retry parameters
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Alert callback
        self.alert_callback: Optional[Callable[[dict], None]] = None
        
        # Create cache directory if needed
        if self.enable_cache and not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)
        
        logger.info(f"Initialized MarketMonitoringAgent for {self.ticker}")
    
    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetches raw OHLCV data from market via Yahoo Finance API.
        Uses retry mechanism and caching.
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            
        Raises:
            Exception: If failed to get data after all attempts
        """
        # Check cache
        if self.enable_cache:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                logger.info(f"Loaded data from cache for {self.ticker}")
                self.raw_data = cached_data.copy()
                return cached_data
        
        # Retry mechanism
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Loading data for {self.ticker} (period={self.period}, interval={self.interval}, attempt {attempt + 1})")
                
                # Load data via yfinance
                data = yf.download(
                    tickers=self.ticker,
                    period=self.period,
                    interval=self.interval,
                    progress=False,
                    auto_adjust=False
                )
                
                # Check for empty data
                if data.empty:
                    raise ValueError(f"Failed to get data for ticker {self.ticker}")
                
                # If data is MultiIndex, convert it
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                # Remove 'Price' column if present
                if 'Price' in data.columns:
                    data = data.drop(columns=['Price'])
                
                # Validate data
                if not self.validate_dataframe(data):
                    raise ValueError(f"Data validation failed for {self.ticker}")
                
                # Ensure all required columns are present
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Save to cache
                if self.enable_cache:
                    self._save_to_cache(data)
                
                # Save raw data
                self.raw_data = data.copy()
                
                logger.info(f"Successfully loaded {len(data)} records")
                return data
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Error loading data (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
        
        # If all attempts failed, try loading from cache
        if self.enable_cache:
            cached_data = self._load_from_cache(ignore_ttl=True)
            if cached_data is not None:
                logger.warning(f"Using stale data from cache for {self.ticker}")
                self.raw_data = cached_data.copy()
                return cached_data
        
        # If nothing helped, raise exception
        logger.error(f"Failed to load data for {self.ticker} after {self.max_retries} attempts")
        raise last_exception or Exception(f"Failed to get data for ticker {self.ticker}")
    
    def _load_from_cache(self, ignore_ttl: bool = False) -> Optional[pd.DataFrame]:
        """Loads data from cache."""
        # Try parquet first, then CSV
        cache_file_parquet = os.path.join(self.cache_path, f"{self.ticker}_{self.interval}_{self.period}.parquet")
        cache_file_csv = os.path.join(self.cache_path, f"{self.ticker}_{self.interval}_{self.period}.csv")
        
        cache_file = None
        use_parquet = False
        
        if os.path.exists(cache_file_parquet):
            cache_file = cache_file_parquet
            use_parquet = True
        elif os.path.exists(cache_file_csv):
            cache_file = cache_file_csv
            use_parquet = False
        else:
            return None
        
        try:
            # Check TTL
            if not ignore_ttl:
                file_time = os.path.getmtime(cache_file)
                if time.time() - file_time > self.cache_ttl:
                    return None
            
            if use_parquet:
                try:
                    data = pd.read_parquet(cache_file)
                    return data
                except Exception:
                    # Fallback to CSV if parquet doesn't work
                    if os.path.exists(cache_file_csv):
                        data = pd.read_csv(cache_file_csv, index_col=0, parse_dates=True)
                        return data
                    return None
            else:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return data
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame) -> None:
        """Saves data to cache."""
        try:
            # Try saving to parquet, if fails - use CSV
            cache_file_parquet = os.path.join(self.cache_path, f"{self.ticker}_{self.interval}_{self.period}.parquet")
            cache_file_csv = os.path.join(self.cache_path, f"{self.ticker}_{self.interval}_{self.period}.csv")
            
            try:
                data.to_parquet(cache_file_parquet)
            except Exception:
                # Fallback to CSV if parquet unavailable
                data.to_csv(cache_file_csv)
                logger.debug(f"Data saved to CSV (parquet unavailable)")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validates DataFrame for required columns and data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                return False
            if df[col].isna().all():
                return False
        
        return True
    
    def validate_schema(self, df: pd.DataFrame) -> List[str]:
        """
        Checks DataFrame schema and returns list of issues.
        
        Args:
            df: DataFrame to check
            
        Returns:
            List of strings with issue descriptions (empty if all OK)
        """
        issues = []
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing column: {col}")
            elif df[col].isna().all():
                issues.append(f"Column {col} is completely empty")
        
        return issues
    
    def compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators based on OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            logger.info("Computing technical indicators...")
            
            df = data.copy()
            
            # Base indicators (always computed)
            if "sma" in self.indicators or "all" in self.indicators:
                df['sma10'] = df['Close'].rolling(window=10, min_periods=1).mean()
                df['sma20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            
            if "rsi" in self.indicators or "all" in self.indicators:
                df['rsi14'] = self._calculate_rsi(df['Close'], period=14)
            
            df['price_change'] = df['Close'].pct_change() * 100
            df['volatility'] = df['Close'].rolling(window=10, min_periods=1).std()
            
            # MACD
            if "macd" in self.indicators or "all" in self.indicators:
                macd_data = self._compute_macd(df['Close'])
                df = pd.concat([df, macd_data], axis=1)
            
            # Bollinger Bands
            if "bb" in self.indicators or "all" in self.indicators:
                bb_data = self._compute_bollinger_bands(df['Close'])
                df = pd.concat([df, bb_data], axis=1)
            
            logger.info("Indicators successfully computed")
            return df
            
        except Exception as e:
            logger.error(f"Error computing indicators: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Computes Relative Strength Index (RSI)."""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rsi = pd.Series(index=prices.index, dtype=float)
        
        rsi[(loss == 0) & (gain > 0)] = 100
        rsi[(gain == 0) & (loss > 0)] = 0
        rsi[(gain == 0) & (loss == 0)] = 50
        
        mask = (gain > 0) & (loss > 0)
        if mask.any():
            rs = gain[mask] / loss[mask]
            rsi[mask] = 100 - (100 / (1 + rs))
        
        rsi = rsi.fillna(50)
        return rsi
    
    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Computes MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist
        })
    
    def _compute_bollinger_bands(self, prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
        """Computes Bollinger Bands."""
        bb_mid = prices.rolling(window=window, min_periods=1).mean()
        bb_std = prices.rolling(window=window, min_periods=1).std()
        bb_upper = bb_mid + (bb_std * n_std)
        bb_lower = bb_mid - (bb_std * n_std)
        
        return pd.DataFrame({
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_mid': bb_mid
        })
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses data: removes NaN, normalizes if needed."""
        try:
            logger.info("Preprocessing data...")
            
            df = data.copy()
            df = df.dropna(how='all')
            
            # Fill NaN for indicators
            if 'sma10' in df.columns:
                df['sma10'] = df['sma10'].fillna(df['Close'])
            if 'sma20' in df.columns:
                df['sma20'] = df['sma20'].fillna(df['Close'])
            if 'rsi14' in df.columns:
                df['rsi14'] = df['rsi14'].fillna(50)
            
            df['price_change'] = df['price_change'].fillna(0)
            df['volatility'] = df['volatility'].fillna(0)
            
            # Fill NaN for MACD and BB
            if 'macd' in df.columns:
                df['macd'] = df['macd'].fillna(0)
                df['macd_signal'] = df['macd_signal'].fillna(0)
                df['macd_hist'] = df['macd_hist'].fillna(0)
            
            if 'bb_upper' in df.columns:
                df['bb_upper'] = df['bb_upper'].fillna(df['Close'])
                df['bb_lower'] = df['bb_lower'].fillna(df['Close'])
                df['bb_mid'] = df['bb_mid'].fillna(df['Close'])
            
            df = df.dropna()
            df.columns = df.columns.str.lower()
            
            # Process timestamp
            if 'timestamp' not in df.columns:
                index_name = df.index.name
                df = df.reset_index()
                
                if index_name and index_name in df.columns and 'price' in str(index_name).lower():
                    df = df.drop(columns=[index_name])
                
                if 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'timestamp'})
                elif 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'timestamp'})
                elif 'date' in df.columns:
                    df = df.rename(columns={'date': 'timestamp'})
                elif 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'timestamp'})
                else:
                    df['timestamp'] = df.index
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Remove 'price' column if present
            for col in list(df.columns):
                if col.lower() == 'price' and col != 'price_change':
                    df = df.drop(columns=[col])
            
            # Select needed columns
            base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            indicator_columns = []
            
            if 'sma10' in df.columns:
                indicator_columns.extend(['sma10', 'sma20'])
            if 'rsi14' in df.columns:
                indicator_columns.append('rsi14')
            if 'macd' in df.columns:
                indicator_columns.extend(['macd', 'macd_signal', 'macd_hist'])
            if 'bb_upper' in df.columns:
                indicator_columns.extend(['bb_upper', 'bb_lower', 'bb_mid'])
            
            indicator_columns.extend(['price_change', 'volatility'])
            
            required_columns = base_columns + indicator_columns
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
            
            if df.index.name and 'price' in str(df.index.name).lower():
                df.index.name = None
            
            logger.info(f"Preprocessing completed. {len(df)} records remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
        """
        Analyzes market conditions and returns structured insights.
        
        Args:
            data: DataFrame with processed data and indicators
            
        Returns:
            dict with keys: trend, signals, strength
        """
        if data.empty:
            return {"trend": "unknown", "signals": {}, "strength": 0.0}
        
        latest = data.iloc[-1]
        
        # Determine trend
        trend = "sideways"
        if 'sma10' in data.columns and 'sma20' in data.columns:
            if latest['sma10'] > latest['sma20']:
                trend = "bull"
            elif latest['sma10'] < latest['sma20']:
                trend = "bear"
        
        # Signals
        signals = {}
        
        # SMA crossover
        if 'sma10' in data.columns and 'sma20' in data.columns and len(data) > 1:
            prev = data.iloc[-2]
            signals['sma_cross'] = bool(
                (prev['sma10'] <= prev['sma20'] and latest['sma10'] > latest['sma20']) or
                (prev['sma10'] >= prev['sma20'] and latest['sma10'] < latest['sma20'])
            )
        else:
            signals['sma_cross'] = False
        
        # RSI state
        if 'rsi14' in data.columns:
            rsi = latest['rsi14']
            if rsi > 70:
                signals['rsi_state'] = "overbought"
            elif rsi < 30:
                signals['rsi_state'] = "oversold"
            else:
                signals['rsi_state'] = "neutral"
        else:
            signals['rsi_state'] = "neutral"
        
        # Trend strength (0.0 - 1.0)
        strength = 0.5
        if 'rsi14' in data.columns:
            rsi = latest['rsi14']
            # The further RSI is from 50, the stronger the trend
            strength = abs(rsi - 50) / 50
        
        # Convert bool to int for JSON serialization
        signals_serializable = {}
        for key, value in signals.items():
            if isinstance(value, bool):
                signals_serializable[key] = int(value)
            else:
                signals_serializable[key] = value
        
        return {
            "trend": trend,
            "signals": signals_serializable,
            "strength": float(strength)
        }
    
    def detect_anomalies(self, data: pd.DataFrame, window: int = 10, z_thresh: float = 3.0) -> List[Dict]:
        """
        Detects anomalies in data.
        
        Args:
            data: DataFrame with data
            window: Window for computing statistics
            z_thresh: Z-score threshold for anomaly detection
            
        Returns:
            List of dictionaries with anomaly information
        """
        anomalies = []
        
        if len(data) < window:
            return anomalies
        
        # Price anomalies
        if 'close' in data.columns:
            prices = data['close']
            mean = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            z_scores = (prices - mean) / std
            
            price_anomalies = data[abs(z_scores) > z_thresh]
            for idx, row in price_anomalies.iterrows():
                timestamp = row.get('timestamp', idx) if 'timestamp' in row else idx
                anomalies.append({
                    "timestamp": str(timestamp),
                    "type": "price_spike",
                    "value": float(row['close']),
                    "z_score": float(z_scores.loc[idx])
                })
        
        # Volume anomalies
        if 'volume' in data.columns:
            volumes = data['volume']
            mean = volumes.rolling(window=window).mean()
            std = volumes.rolling(window=window).std()
            z_scores = (volumes - mean) / std
            
            volume_anomalies = data[abs(z_scores) > z_thresh]
            for idx, row in volume_anomalies.iterrows():
                timestamp = row.get('timestamp', idx) if 'timestamp' in row else idx
                anomalies.append({
                    "timestamp": str(timestamp),
                    "type": "volume_spike",
                    "value": float(row['volume']),
                    "z_score": float(z_scores.loc[idx])
                })
        
        return anomalies
    
    def generate_insights(self, data: pd.DataFrame, analysis_dict: Dict) -> Dict:
        """
        Generates text insights about market state.
        
        Args:
            data: DataFrame with data
            analysis_dict: Result from analyze_market_conditions()
            
        Returns:
            dict with fields summary and metrics
        """
        if data.empty:
            return {
                "summary": "Insufficient data for analysis",
                "metrics": {}
            }
        
        latest = data.iloc[-1]
        
        # Form summary
        trend = analysis_dict.get("trend", "unknown")
        strength = analysis_dict.get("strength", 0.0)
        signals = analysis_dict.get("signals", {})
        
        summary_parts = [f"Trend: {trend.upper()}"]
        
        if 'rsi_state' in signals:
            summary_parts.append(f"RSI: {signals['rsi_state']}")
        
        if strength > 0.7:
            summary_parts.append("Strong trend")
        elif strength < 0.3:
            summary_parts.append("Weak trend")
        
        summary = ". ".join(summary_parts)
        
        # Metrics
        metrics = {
            "current_price": float(latest['close']) if 'close' in latest else None,
            "trend": trend,
            "strength": strength,
            "rsi": float(latest['rsi14']) if 'rsi14' in latest else None
        }
        
        return {
            "summary": summary,
            "metrics": metrics
        }
    
    def should_alert(self, thresholds: Dict) -> Optional[Dict]:
        """
        Checks if alert should be sent based on threshold values.
        
        Args:
            thresholds: dict with thresholds, e.g.:
                {"rsi_overbought": 70, "rsi_oversold": 30, "volatility": 0.02}
        
        Returns:
            dict with alert information or None if alert not needed
        """
        if self.processed_data is None or self.processed_data.empty:
            return None
        
        latest = self.processed_data.iloc[-1]
        alert_info = {}
        
        # Check RSI
        if 'rsi14' in latest:
            rsi = latest['rsi14']
            if 'rsi_overbought' in thresholds and rsi > thresholds['rsi_overbought']:
                alert_info['rsi_overbought'] = rsi
            if 'rsi_oversold' in thresholds and rsi < thresholds['rsi_oversold']:
                alert_info['rsi_oversold'] = rsi
        
        # Check volatility
        if 'volatility' in latest:
            vol = latest['volatility']
            if 'volatility' in thresholds and vol > thresholds['volatility']:
                alert_info['high_volatility'] = vol
        
        if alert_info:
            return {
                "ticker": self.ticker,
                "timestamp": str(latest.get('timestamp', datetime.now())),
                "alerts": alert_info
            }
        
        return None
    
    def on_alert(self, callback_fn: Callable[[dict], None]) -> None:
        """
        Sets callback function for handling alerts.
        
        Args:
            callback_fn: Function to be called on alert
                        Signature: Callable[[dict], None]
        """
        self.alert_callback = callback_fn
        logger.info(f"Alert callback set for {self.ticker}")
    
    def monitor_continuously(
        self,
        interval_seconds: int = 300,
        stop_event: Optional[threading.Event] = None,
        thresholds: Optional[Dict] = None
    ) -> None:
        """
        Starts continuous monitoring with specified interval.
        
        Args:
            interval_seconds: Interval between updates in seconds
            stop_event: threading.Event for stopping monitoring
            thresholds: Threshold values for alerts
        """
        if stop_event is None:
            stop_event = threading.Event()
        
        logger.info(f"Starting continuous monitoring for {self.ticker} (interval: {interval_seconds}s)")
        
        while not stop_event.is_set():
            try:
                # Get and process data
                data = self.get_processed_data(analyze=True)
                
                # Save state
                self.save_state(data[0] if isinstance(data, tuple) else data)
                
                # Check alerts
                if thresholds:
                    alert = self.should_alert(thresholds)
                    if alert and self.alert_callback:
                        self.alert_callback(alert)
                
                # Wait until next update
                stop_event.wait(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                stop_event.wait(interval_seconds)
    
    def save_state(self, processed_df: pd.DataFrame) -> None:
        """Saves state snapshot to history."""
        self.state_history.append(processed_df.copy())
        logger.debug(f"State saved to history (size: {len(self.state_history)})")
    
    def get_history(self) -> List[pd.DataFrame]:
        """Returns state history."""
        return list(self.state_history)
    
    def get_processed_data(self, analyze: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Main method: fetches and processes data.
        
        Args:
            analyze: If True, also performs analysis and returns tuple (df, analysis_dict)
        
        Returns:
            DataFrame or tuple (DataFrame, analysis_dict) if analyze=True
        """
        try:
            logger.info(f"Starting data processing for {self.ticker}")
            
            raw_data = self.fetch_raw_data()
            data_with_indicators = self.compute_indicators(raw_data)
            processed_data = self.preprocess(data_with_indicators)
            
            self.processed_data = processed_data.copy()
            
            if analyze:
                analysis = self.analyze_market_conditions(processed_data)
                logger.info(f"Processing completed successfully. Got {len(processed_data)} records")
                return processed_data, analysis
            
            logger.info(f"Processing completed successfully. Got {len(processed_data)} records")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in get_processed_data: {str(e)}")
            raise
    
    def send_to_decision_agent(
        self,
        transport: str = "direct",
        endpoint: Optional[str] = None
    ) -> Dict:
        """
        Sends data to Decision Agent in standardized format.
        
        Args:
            transport: Transport method ("direct", "http", "mq")
            endpoint: URL for HTTP transport
        
        Returns:
            dict with data in standardized format
        """
        if self.processed_data is None or self.processed_data.empty:
            raise ValueError("No processed data. Call get_processed_data() first.")
        
        # Get analysis
        analysis = self.analyze_market_conditions(self.processed_data)
        latest = self.processed_data.iloc[-1]
        
        # Form standardized message
        message = {
            "timestamp": datetime.now().isoformat() + "Z",
            "ticker": self.ticker,
            "ohlcv": {
                "open": float(latest['open']),
                "high": float(latest['high']),
                "low": float(latest['low']),
                "close": float(latest['close']),
                "volume": int(latest['volume'])
            },
            "indicators": {},
            "analysis": analysis,
            "meta": {
                "source": "yfinance",
                "fetched_at": datetime.now().isoformat() + "Z"
            }
        }
        
        # Add indicators
        for col in ['sma10', 'sma20', 'rsi14', 'price_change', 'volatility',
                   'macd', 'macd_signal', 'macd_hist',
                   'bb_upper', 'bb_lower', 'bb_mid']:
            if col in latest:
                message["indicators"][col] = float(latest[col])
        
        # Handle transport
        if transport == "direct":
            return message
        elif transport == "http":
            if endpoint is None:
                raise ValueError("endpoint required for HTTP transport")
            # Real HTTP sending can be added here
            logger.info(f"Sending data to {endpoint} (stub)")
            return message
        elif transport == "mq":
            logger.info("Sending via message queue (stub)")
            return message
        else:
            raise ValueError(f"Unknown transport: {transport}")
    
    def handle_request(self, request: Dict) -> Dict:
        """
        Handles requests from other agents.
        
        Args:
            request: dict with request, e.g.:
                {"type": "get_latest", "n": 10}
                {"type": "get_indicator", "indicator": "rsi14"}
        
        Returns:
            dict with response
        """
        if self.processed_data is None:
            return {"error": "No data. Call get_processed_data() first."}
        
        request_type = request.get("type")
        
        if request_type == "get_latest":
            n = request.get("n", 1)
            data = self.processed_data.tail(n).to_dict('records')
            return {"data": data}
        
        elif request_type == "get_indicator":
            indicator = request.get("indicator")
            if indicator in self.processed_data.columns:
                return {"indicator": indicator, "values": self.processed_data[indicator].tolist()}
            else:
                return {"error": f"Indicator {indicator} not found"}
        
        elif request_type == "get_analysis":
            analysis = self.analyze_market_conditions(self.processed_data)
            return {"analysis": analysis}
        
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    @staticmethod
    def process_batch(
        tickers: List[str],
        interval: str = "1h",
        period: str = "1mo",
        parallel: bool = False,
        max_workers: int = 4
    ) -> Dict[str, pd.DataFrame]:
        """
        Processes multiple tickers.
        
        Args:
            tickers: List of tickers
            interval: Candle interval
            period: Data period
            parallel: Use parallel processing
            max_workers: Maximum number of threads
        
        Returns:
            dict with tickers as keys and DataFrames as values
        """
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        MarketMonitoringAgent(ticker, interval, period).get_processed_data
                    ): ticker
                    for ticker in tickers
                }
                
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        results[ticker] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {e}")
                        results[ticker] = pd.DataFrame()
        else:
            for ticker in tickers:
                try:
                    agent = MarketMonitoringAgent(ticker, interval, period)
                    results[ticker] = agent.get_processed_data()
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    results[ticker] = pd.DataFrame()
        
        return results


# Example usage
if __name__ == "__main__":
    agent = MarketMonitoringAgent(ticker="AAPL", interval="1h", period="1mo")
    
    try:
        data, analysis = agent.get_processed_data(analyze=True)
        print("\n" + "="*80)
        print("PROCESSED DATA:")
        print("="*80)
        print(data.head(10))
        print("\nANALYSIS:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        print("="*80)
    except Exception as e:
        print(f"Error: {e}")
