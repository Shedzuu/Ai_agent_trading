# Руководство по использованию Multi-Agent Trading System

## Обзор системы

Система состоит из 3 агентов, которые работают вместе для автоматической торговли:

1. **Market Monitoring Agent** - мониторит рынок и собирает данные
2. **Decision Making Agent** - принимает решения BUY/SELL/HOLD используя AI модель
3. **Execution Agent** - выполняет сделки и записывает результаты

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### Пример 1: Полный рабочий процесс

```python
from agents.coordinator import AgentCoordinator

# Создать координатор для тикера AAPL
coordinator = AgentCoordinator(ticker="AAPL", auto_start=False)

# Запустить один цикл торговли
result = coordinator.run_single_cycle()

print(f"Решение: {result['decision']['action']}")
print(f"Уверенность: {result['decision']['confidence']:.2f}")
print(f"Статус выполнения: {result['execution']['status']}")
```

### Пример 2: Пошаговый процесс

```python
from agents.market_monitor import MarketMonitoringAgent
from agents.decision_maker import DecisionMakingAgent
from agents.execution_agent import ExecutionAgent

# Шаг 1: Получить данные рынка
market_agent = MarketMonitoringAgent(ticker="AAPL", interval="1h", period="1mo")
market_data = market_agent.send_to_decision_agent(transport="direct")

# Шаг 2: Принять решение
decision_agent = DecisionMakingAgent(model_type="random_forest")
decision = decision_agent.receive_market_data(market_data)

# Шаг 3: Выполнить сделку
execution_agent = ExecutionAgent(execution_mode="simulated")
result = execution_agent.receive_decision(decision)
```

## Использование Backend API

### Запуск сервера

```bash
python -m api.server
```

Или используя uvicorn напрямую:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Получить данные рынка

```bash
GET /api/market/data/{ticker}?interval=1h&period=1mo
```

Или POST:

```bash
POST /api/market/data
{
  "ticker": "AAPL",
  "interval": "1h",
  "period": "1mo"
}
```

#### 2. Принять решение

```bash
POST /api/decision/make
{
  "market_data": { ... }  # Данные от Market Agent
}
```

#### 3. Выполнить сделку

```bash
POST /api/execution/execute
{
  "decision": { ... }  # Решение от Decision Agent
}
```

#### 4. Полный рабочий процесс

```bash
POST /api/workflow/run
{
  "ticker": "AAPL",
  "interval": "1h",
  "period": "1mo"
}
```

#### 5. Запустить координатор

```bash
POST /api/coordinator/start
{
  "ticker": "AAPL",
  "interval_seconds": 300
}
```

#### 6. Получить статус системы

```bash
GET /api/status
```

## Примеры использования

Запустите полные примеры:

```bash
python example_full_workflow.py
```

## Архитектура системы

```
┌─────────────────────┐
│ MarketMonitoringAgent│
│    (Agent 1)         │
│                      │
│ • Fetches data       │
│ • Computes indicators│
│ • Analyzes conditions│
└──────────┬───────────┘
           │
           │ Sends: JSON message with OHLCV, indicators, analysis
           ↓
┌─────────────────────┐
│  DecisionMakingAgent│
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
│ ExecutionAgent      │
│    (Agent 3)        │
│                      │
│ • Executes trades    │
│ • Manages orders     │
│ • Reports results    │
└─────────────────────┘
```

## Настройка агентов

### Market Monitoring Agent

```python
market_agent = MarketMonitoringAgent(
    ticker="AAPL",
    interval="1h",      # 1h, 30m, 1d, 5m, 15m
    period="1mo",      # 1mo, 3mo, 1y, 6mo, 1d
    enable_cache=True,
    indicators=["sma", "rsi", "macd", "bb"]
)
```

### Decision Making Agent

```python
decision_agent = DecisionMakingAgent(
    model_type="random_forest",  # "random_forest", "gradient_boosting", "rule_based"
    risk_tolerance="medium",     # "low", "medium", "high"
    max_position_size=0.1,       # Max 10% of portfolio per trade
    min_confidence=0.6,          # Minimum confidence for action
    enable_ai=True
)
```

### Execution Agent

```python
execution_agent = ExecutionAgent(
    execution_mode="simulated",  # "simulated" or "real"
    enable_slippage=True,
    slippage_factor=0.001,       # 0.1% slippage
    commission_rate=0.001        # 0.1% commission
)
```

## Коммуникация между агентами

### Формат данных от Market Agent

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
    "macd_hist": -0.7,
    "volatility": 1.2
  },
  "analysis": {
    "trend": "bull",
    "signals": {
      "sma_cross": 0,
      "rsi_state": "overbought"
    },
    "strength": 0.72
  }
}
```

### Формат решения от Decision Agent

```json
{
  "action": "BUY",
  "ticker": "AAPL",
  "confidence": 0.75,
  "reasoning": "BUY decision (confidence: 0.75). Trend: BULL. RSI oversold (45.0). Strong trend (strength: 0.65)",
  "quantity": 10,
  "price": 151.0,
  "timestamp": "2025-12-05T12:00:00Z",
  "risk_score": 0.3,
  "stop_loss": 147.97,
  "take_profit": 163.08
}
```

### Формат результата выполнения

```json
{
  "status": "executed",
  "order_id": "ORD_20251205120000123456",
  "ticker": "AAPL",
  "action": "BUY",
  "quantity": 10,
  "executed_price": 151.15,
  "commission": 15.12,
  "slippage": 0.15,
  "timestamp": "2025-12-05T12:00:00Z"
}
```

## AI Модель

Decision Making Agent использует AI модель (Random Forest или Gradient Boosting) для принятия решений.

Модель обучается на:
- Технических индикаторах (RSI, MACD, SMA, Bollinger Bands)
- Анализе тренда
- Сигналах рынка

Если AI недоступна, используется правило-основанная логика.

## Управление рисками

Система включает управление рисками:

- **Максимальный размер позиции**: ограничение на размер каждой сделки
- **Stop Loss**: автоматическая остановка убытков
- **Take Profit**: автоматическая фиксация прибыли
- **Минимальная уверенность**: сделки выполняются только при достаточной уверенности
- **Оценка риска**: расчет риска на основе волатильности

## Логирование сделок

Все сделки записываются в файл `trades_log.json`:

```json
[
  {
    "order_id": "ORD_20251205120000123456",
    "ticker": "AAPL",
    "action": "BUY",
    "quantity": 10,
    "executed_price": 151.15,
    "commission": 15.12,
    "timestamp": "2025-12-05T12:00:00Z",
    "status": "executed"
  }
]
```

## Непрерывный мониторинг

Для непрерывного мониторинга используйте Coordinator:

```python
coordinator = AgentCoordinator(
    ticker="AAPL",
    auto_start=True  # Автоматически запустить мониторинг
)

# Остановить мониторинг
coordinator.stop_monitoring()
```

## Интеграция с фронтендом

Backend API готов для интеграции с фронтендом. Используйте следующие endpoints:

- `GET /api/status` - статус системы
- `GET /api/market/data/{ticker}` - данные рынка
- `POST /api/workflow/run` - запуск полного цикла
- `GET /api/execution/history` - история сделок

## Troubleshooting

### Проблема: Не удается получить данные рынка

**Решение**: Проверьте интернет-соединение и доступность Yahoo Finance API. Используйте кэш для офлайн работы.

### Проблема: AI модель не обучается

**Решение**: Убедитесь, что установлен scikit-learn: `pip install scikit-learn`

### Проблема: Ошибки при выполнении сделок

**Решение**: Проверьте, что Execution Agent использует режим "simulated" для тестирования.

## Дополнительная информация

См. `README.md` для более подробной информации о каждом агенте.

