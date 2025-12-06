# Руководство по тестированию системы

## 🚀 Быстрый старт

### Вариант 1: Автоматическое тестирование (рекомендуется)

Запустите тестовый скрипт, который проверит все компоненты:

```bash
python test_system.py
```

Этот скрипт:
- ✅ Проверит Market Monitoring Agent
- ✅ Проверит Decision Making Agent (обучит AI модель)
- ✅ Проверит Execution Agent
- ✅ Проверит полный workflow через Coordinator
- ✅ Покажет инструкции для тестирования API

---

## 📋 Вариант 2: Ручное тестирование компонентов

### Тест 1: Market Monitoring Agent

```python
from agents.market_monitor import MarketMonitoringAgent

# Создать агента
agent = MarketMonitoringAgent(ticker="AAPL", interval="1d", period="1mo")

# Получить данные
data, analysis = agent.get_processed_data(analyze=True)

print(f"Получено записей: {len(data)}")
print(f"Тренд: {analysis['trend']}")
print(f"Сила: {analysis['strength']:.2f}")

# Подготовить данные для Decision Agent
market_message = agent.send_to_decision_agent(transport="direct")
print(f"Цена: ${market_message['ohlcv']['close']:.2f}")
```

**Ожидаемый результат:**
- ✓ Данные получены (не пустой DataFrame)
- ✓ Анализ содержит trend, signals, strength
- ✓ market_message в правильном JSON формате

---

### Тест 2: Decision Making Agent

```python
from agents.decision_maker import DecisionMakingAgent
from agents.market_monitor import MarketMonitoringAgent

# Получить данные от Market Agent
market_agent = MarketMonitoringAgent(ticker="AAPL")
market_data = market_agent.send_to_decision_agent(transport="direct")

# Создать Decision Agent
decision_agent = DecisionMakingAgent(
    model_type="random_forest",
    risk_tolerance="medium"
)

# Принять решение
decision = decision_agent.receive_market_data(market_data)

print(f"Действие: {decision['action']}")
print(f"Уверенность: {decision['confidence']:.2f}")
print(f"Обоснование: {decision['reasoning']}")
```

**Ожидаемый результат:**
- ✓ action = "BUY", "SELL", или "HOLD"
- ✓ confidence между 0.0 и 1.0
- ✓ reasoning содержит объяснение
- ✓ Модель обучена (если enable_ai=True)

**Примечание:** При первом запуске модель будет обучена (может занять несколько секунд).

---

### Тест 3: Execution Agent

```python
from agents.execution_agent import ExecutionAgent
from agents.decision_maker import DecisionMakingAgent
from agents.market_monitor import MarketMonitoringAgent

# Получить решение
market_agent = MarketMonitoringAgent(ticker="AAPL")
market_data = market_agent.send_to_decision_agent(transport="direct")

decision_agent = DecisionMakingAgent()
decision = decision_agent.receive_market_data(market_data)

# Выполнить сделку
execution_agent = ExecutionAgent(execution_mode="simulated")
result = execution_agent.receive_decision(decision)

print(f"Статус: {result['status']}")
if result['status'] == 'executed':
    print(f"Order ID: {result['order_id']}")
    print(f"Цена: ${result['executed_price']:.2f}")
    print(f"Комиссия: ${result['commission']:.2f}")
```

**Ожидаемый результат:**
- ✓ status = "executed", "hold", или "rejected"
- ✓ Если executed: order_id, executed_price, commission присутствуют
- ✓ Сделка записана в trades_log.json

---

### Тест 4: Полный Workflow (Coordinator)

```python
from agents.coordinator import AgentCoordinator

# Создать coordinator
coordinator = AgentCoordinator(ticker="AAPL", auto_start=False)

# Запустить один цикл
result = coordinator.run_single_cycle()

print(f"Решение: {result['decision']['action']}")
print(f"Статус: {result['execution']['status']}")
print(f"Портфель: ${result['portfolio']['cash']:.2f}")
```

**Ожидаемый результат:**
- ✓ Полный цикл выполнен без ошибок
- ✓ Все три агента работают вместе
- ✓ Результат содержит market_data, decision, execution, portfolio

---

## 🌐 Тестирование Backend API

### Шаг 1: Запустить сервер

В одном терминале:

```bash
python -m api.server
```

Или:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Сервер запустится на `http://localhost:8000`

### Шаг 2: Проверить здоровье сервера

```bash
curl http://localhost:8000/api/health
```

**Ожидаемый результат:**
```json
{"status": "healthy", "timestamp": "2025-12-05T..."}
```

### Шаг 3: Получить данные рынка

```bash
curl http://localhost:8000/api/market/data/AAPL
```

**Ожидаемый результат:**
```json
{
  "success": true,
  "ticker": "AAPL",
  "data_points": 30,
  "market_data": {...},
  "analysis": {...}
}
```

### Шаг 4: Запустить полный workflow

```bash
curl -X POST http://localhost:8000/api/workflow/run \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "interval": "1d", "period": "1mo"}'
```

**Ожидаемый результат:**
```json
{
  "success": true,
  "workflow": {
    "ticker": "AAPL",
    "market_data": {...},
    "decision": {...},
    "execution": {...},
    "portfolio": {...}
  }
}
```

### Шаг 5: Использовать Swagger UI

Откройте в браузере:
```
http://localhost:8000/docs
```

Это интерактивный интерфейс для тестирования всех endpoints.

---

## 🧪 Запуск примеров

### Пример 1: Базовое использование

```bash
python example_usage.py
```

Показывает возможности Market Monitoring Agent.

### Пример 2: Полный workflow

```bash
python example_full_workflow.py
```

Демонстрирует:
- Ручной workflow
- Coordinator workflow
- Обработку нескольких тикеров
- Историю решений и выполнений

---

## ✅ Чек-лист тестирования

### Базовые тесты

- [ ] Market Agent получает данные
- [ ] Market Agent вычисляет индикаторы
- [ ] Market Agent отправляет данные в правильном формате
- [ ] Decision Agent принимает данные
- [ ] Decision Agent обучает модель (при первом запуске)
- [ ] Decision Agent принимает решения (BUY/SELL/HOLD)
- [ ] Execution Agent выполняет сделки
- [ ] Execution Agent записывает в лог
- [ ] Coordinator управляет полным циклом

### Интеграционные тесты

- [ ] Все агенты работают вместе
- [ ] Данные передаются корректно между агентами
- [ ] Портфель обновляется после сделок
- [ ] История сохраняется

### API тесты

- [ ] Сервер запускается
- [ ] Health check работает
- [ ] Market data endpoint работает
- [ ] Decision endpoint работает
- [ ] Execution endpoint работает
- [ ] Workflow endpoint работает
- [ ] Coordinator endpoints работают

---

## 🐛 Решение проблем

### Проблема: "ModuleNotFoundError: No module named 'sklearn'"

**Решение:**
```bash
pip install scikit-learn
```

### Проблема: "Failed to get data for ticker"

**Решение:**
- Проверьте интернет-соединение
- Проверьте правильность тикера
- Попробуйте другой тикер (AAPL, MSFT, GOOGL)

### Проблема: "Model not trained"

**Решение:**
- Это нормально при первом запуске
- Модель обучится автоматически
- Подождите несколько секунд

### Проблема: "API server not responding"

**Решение:**
- Убедитесь, что сервер запущен
- Проверьте порт (по умолчанию 8000)
- Проверьте firewall настройки

---

## 📊 Проверка результатов

### Проверка логов сделок

```python
import json

with open('trades_log.json', 'r') as f:
    trades = json.load(f)

print(f"Всего сделок: {len(trades)}")
for trade in trades[-5:]:  # Последние 5
    print(f"{trade['timestamp']}: {trade['action']} {trade['quantity']} {trade['ticker']}")
```

### Проверка портфеля

```python
from agents.decision_maker import DecisionMakingAgent

agent = DecisionMakingAgent()
portfolio = agent.get_portfolio_status()

print(f"Денежные средства: ${portfolio['cash']:.2f}")
print(f"Позиции: {portfolio['positions']}")
```

### Проверка истории решений

```python
from agents.decision_maker import DecisionMakingAgent

agent = DecisionMakingAgent()
# ... выполнить несколько решений ...

history = agent.get_decision_history(n=10)
for decision in history:
    print(f"{decision['timestamp']}: {decision['action']} (confidence: {decision['confidence']:.2f})")
```

---

## 🎯 Рекомендуемый порядок тестирования

1. **Запустите `test_system.py`** - автоматическая проверка всех компонентов
2. **Запустите `example_full_workflow.py`** - посмотрите примеры использования
3. **Запустите API сервер** - проверьте веб-интерфейс
4. **Протестируйте через Swagger UI** - интерактивное тестирование API

---

## 📝 Заметки

- При первом запуске Decision Agent обучит модель (может занять 5-10 секунд)
- Market Agent использует кэш для ускорения (данные кэшируются на 1 час)
- Execution Agent работает в симулированном режиме (не реальные сделки)
- Все сделки записываются в `trades_log.json`

---

Готово! Теперь вы можете протестировать всю систему. 🚀

