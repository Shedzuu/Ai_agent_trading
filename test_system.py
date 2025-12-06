"""
Тестовый скрипт для проверки работы Multi-Agent Trading System

Запустите этот скрипт для проверки всех компонентов системы.
"""

import sys
import json
from datetime import datetime

def print_header(title):
    """Печатает заголовок секции."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_market_agent():
    """Тест 1: Market Monitoring Agent"""
    print_header("ТЕСТ 1: Market Monitoring Agent")
    
    try:
        from agents.market_monitor import MarketMonitoringAgent
        
        print("Создание Market Monitoring Agent для AAPL...")
        agent = MarketMonitoringAgent(
            ticker="AAPL",
            interval="1d",
            period="1mo",
            enable_cache=True
        )
        print("✓ Агент создан")
        
        print("\nПолучение и обработка данных...")
        data, analysis = agent.get_processed_data(analyze=True)
        print(f"✓ Получено {len(data)} записей")
        print(f"✓ Тренд: {analysis['trend']}")
        print(f"✓ Сила тренда: {analysis['strength']:.2f}")
        
        print("\nПодготовка данных для Decision Agent...")
        market_message = agent.send_to_decision_agent(transport="direct")
        print(f"✓ Данные подготовлены")
        print(f"  - Тикер: {market_message['ticker']}")
        print(f"  - Цена: ${market_message['ohlcv']['close']:.2f}")
        print(f"  - RSI: {market_message['indicators'].get('rsi14', 'N/A'):.2f}")
        
        return True, market_message
        
    except Exception as e:
        print(f"✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_decision_agent(market_data):
    """Тест 2: Decision Making Agent"""
    print_header("ТЕСТ 2: Decision Making Agent")
    
    if market_data is None:
        print("⚠ Пропуск теста: нет данных от Market Agent")
        return False, None
    
    try:
        from agents.decision_maker import DecisionMakingAgent
        
        print("Создание Decision Making Agent...")
        agent = DecisionMakingAgent(
            model_type="random_forest",
            risk_tolerance="medium",
            enable_ai=True
        )
        print("✓ Агент создан")
        
        print("\nОбучение AI модели (если нужно)...")
        print("  Это может занять несколько секунд...")
        
        print("\nПринятие решения на основе рыночных данных...")
        decision = agent.receive_market_data(market_data)
        
        print(f"✓ Решение принято:")
        print(f"  - Действие: {decision['action']}")
        print(f"  - Уверенность: {decision['confidence']:.2f}")
        print(f"  - Обоснование: {decision['reasoning']}")
        if decision['action'] != 'HOLD':
            print(f"  - Количество: {decision['quantity']}")
            print(f"  - Цена: ${decision['price']:.2f}")
            if 'stop_loss' in decision:
                print(f"  - Stop Loss: ${decision['stop_loss']:.2f}")
                print(f"  - Take Profit: ${decision['take_profit']:.2f}")
        
        print(f"\n✓ Модель: {decision.get('model_type', 'unknown')}")
        print(f"✓ Оценка риска: {decision.get('risk_score', 0):.2f}")
        
        return True, decision
        
    except Exception as e:
        print(f"✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_execution_agent(decision):
    """Тест 3: Execution Agent"""
    print_header("ТЕСТ 3: Execution Agent")
    
    if decision is None:
        print("⚠ Пропуск теста: нет решения от Decision Agent")
        return False, None
    
    try:
        from agents.execution_agent import ExecutionAgent
        
        print("Создание Execution Agent...")
        agent = ExecutionAgent(execution_mode="simulated")
        print("✓ Агент создан")
        
        print("\nВыполнение сделки...")
        execution_result = agent.receive_decision(decision)
        
        print(f"✓ Результат выполнения:")
        print(f"  - Статус: {execution_result['status']}")
        print(f"  - Сообщение: {execution_result['message']}")
        
        if execution_result['status'] == 'executed':
            print(f"  - Order ID: {execution_result['order_id']}")
            print(f"  - Тикер: {execution_result['ticker']}")
            print(f"  - Действие: {execution_result['action']}")
            print(f"  - Количество: {execution_result['quantity']}")
            print(f"  - Запрошенная цена: ${execution_result['requested_price']:.2f}")
            print(f"  - Выполненная цена: ${execution_result['executed_price']:.2f}")
            print(f"  - Проскальзывание: ${execution_result['slippage']:.4f}")
            print(f"  - Комиссия: ${execution_result['commission']:.2f}")
            print(f"  - Общая стоимость: ${execution_result['total_cost']:.2f}")
        
        print("\nСтатистика сделок:")
        stats = agent.get_trade_statistics()
        print(f"  - Всего сделок: {stats['total_trades']}")
        print(f"  - Покупок: {stats['buy_trades']}")
        print(f"  - Продаж: {stats['sell_trades']}")
        print(f"  - Общий объем: ${stats['total_volume']:.2f}")
        print(f"  - Общая комиссия: ${stats['total_commission']:.2f}")
        
        return True, execution_result
        
    except Exception as e:
        print(f"✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_coordinator():
    """Тест 4: Agent Coordinator (полный workflow)"""
    print_header("ТЕСТ 4: Agent Coordinator (Полный Workflow)")
    
    try:
        from agents.coordinator import AgentCoordinator
        
        print("Создание Coordinator для AAPL...")
        coordinator = AgentCoordinator(
            ticker="AAPL",
            auto_start=False
        )
        print("✓ Coordinator создан")
        
        print("\nЗапуск полного цикла торговли...")
        print("  (Market → Decision → Execution)")
        result = coordinator.run_single_cycle()
        
        print(f"\n✓ Цикл завершен:")
        print(f"  - Тикер: {result['ticker']}")
        print(f"  - Цена рынка: ${result['market_data']['price']:.2f}")
        print(f"  - Тренд: {result['market_data']['trend']}")
        print(f"  - Решение: {result['decision']['action']}")
        print(f"  - Уверенность: {result['decision']['confidence']:.2f}")
        print(f"  - Статус выполнения: {result['execution']['status']}")
        
        if result['execution']['status'] == 'executed':
            print(f"  - Order ID: {result['execution']['order_id']}")
            print(f"  - Выполнено по цене: ${result['execution']['executed_price']:.2f}")
        
        print("\nСтатус портфеля:")
        portfolio = result['portfolio']
        print(f"  - Денежные средства: ${portfolio['cash']:.2f}")
        print(f"  - Позиции: {len(portfolio['positions'])}")
        if portfolio['positions']:
            for ticker, pos in portfolio['positions'].items():
                print(f"    {ticker}: {pos['quantity']} акций @ ${pos['avg_price']:.2f}")
        
        return True, result
        
    except Exception as e:
        print(f"✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_api_server():
    """Тест 5: Backend API Server"""
    print_header("ТЕСТ 5: Backend API Server")
    
    print("⚠ Тест API требует запущенного сервера")
    print("\nДля тестирования API:")
    print("1. Запустите сервер в отдельном терминале:")
    print("   python -m api.server")
    print("   или")
    print("   uvicorn api.server:app --host 0.0.0.0 --port 8000")
    print("\n2. Затем используйте curl или Postman:")
    print("   curl http://localhost:8000/api/health")
    print("   curl http://localhost:8000/api/market/data/AAPL")
    print("\nИли откройте в браузере:")
    print("   http://localhost:8000/docs")
    print("   (Swagger UI для интерактивного тестирования)")
    
    return True

def run_all_tests():
    """Запускает все тесты последовательно."""
    print("\n" + "="*80)
    print("  ТЕСТИРОВАНИЕ MULTI-AGENT TRADING SYSTEM")
    print("="*80)
    
    results = {}
    
    # Тест 1: Market Agent
    success, market_data = test_market_agent()
    results['market_agent'] = success
    
    if not success:
        print("\n⚠ Критическая ошибка в Market Agent. Остальные тесты могут не работать.")
        return results
    
    # Тест 2: Decision Agent
    success, decision = test_decision_agent(market_data)
    results['decision_agent'] = success
    
    # Тест 3: Execution Agent
    success, execution = test_execution_agent(decision)
    results['execution_agent'] = success
    
    # Тест 4: Coordinator
    success, workflow_result = test_coordinator()
    results['coordinator'] = success
    
    # Тест 5: API (информационный)
    test_api_server()
    results['api_info'] = True
    
    # Итоги
    print_header("ИТОГИ ТЕСТИРОВАНИЯ")
    
    total = len([k for k in results.keys() if k != 'api_info'])
    passed = sum(1 for k, v in results.items() if v and k != 'api_info')
    
    print(f"Пройдено тестов: {passed}/{total}")
    print("\nРезультаты:")
    for test_name, result in results.items():
        if test_name != 'api_info':
            status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
            print(f"  {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print(f"\n⚠ {total - passed} тест(ов) провалено. Проверьте ошибки выше.")
    
    return results

if __name__ == "__main__":
    try:
        results = run_all_tests()
        sys.exit(0 if all(v for k, v in results.items() if k != 'api_info') else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Тестирование прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

