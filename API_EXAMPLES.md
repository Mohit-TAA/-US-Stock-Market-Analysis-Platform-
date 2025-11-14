# StockAI Pro - API Examples

## 🔐 Authentication

All API requests require authentication using your API key. Include it in the Authorization header:

```bash
Authorization: Bearer YOUR_API_KEY
```

Get your API key from the Account Settings page.

## 📊 Endpoints

### 1. Get Stock Data

**Endpoint:** `GET /api/v1/stocks/{symbol}`

**Description:** Retrieve historical price data and technical indicators

**Example Request:**

```bash
curl -X GET "https://api.stockaipro.com/v1/stocks/AAPL?period=1y" \
  -H "Authorization: Bearer sk_abc123..."
```

**Example Response:**

```json
{
  "symbol": "AAPL",
  "current_price": 178.45,
  "change": 2.35,
  "change_percent": 1.34,
  "volume": 52847392,
  "market_cap": 2800000000000,
  "data": [
    {
      "date": "2024-01-15",
      "open": 176.50,
      "high": 179.20,
      "low": 176.10,
      "close": 178.45,
      "volume": 52847392,
      "rsi": 58.23,
      "macd": 1.45,
      "bb_upper": 182.30,
      "bb_lower": 174.50
    }
  ]
}
```

**Python Example:**

```python
import requests

API_KEY = "sk_abc123..."
BASE_URL = "https://api.stockaipro.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.get(
    f"{BASE_URL}/stocks/AAPL",
    headers=headers,
    params={"period": "1y"}
)

data = response.json()
print(f"Current Price: ${data['current_price']}")
```

### 2. Get AI Analysis

**Endpoint:** `POST /api/v1/ai/summary`

**Description:** Generate AI-powered stock analysis

**Example Request:**

```bash
curl -X POST "https://api.stockaipro.com/v1/ai/summary" \
  -H "Authorization: Bearer sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "include_fundamentals": true,
    "include_predictions": true
  }'
```

**Example Response:**

```json
{
  "symbol": "AAPL",
  "summary": "Apple Inc. is currently trading at $178.45, showing strong bullish momentum...",
  "sentiment": "Positive",
  "risk_level": "Low",
  "recommendation": "BUY",
  "confidence": 0.85,
  "key_points": [
    "Strong revenue growth (15% YoY)",
    "Positive technical momentum",
    "High institutional ownership"
  ]
}
```

**Python Example:**

```python
response = requests.post(
    f"{BASE_URL}/ai/summary",
    headers=headers,
    json={
        "symbol": "AAPL",
        "include_fundamentals": True,
        "include_predictions": True
    }
)

analysis = response.json()
print(f"Recommendation: {analysis['recommendation']}")
print(f"Summary: {analysis['summary']}")
```

### 3. Get ML Predictions

**Endpoint:** `GET /api/v1/predictions/{symbol}`

**Description:** Get machine learning model predictions

**Example Request:**

```bash
curl -X GET "https://api.stockaipro.com/v1/predictions/TSLA" \
  -H "Authorization: Bearer sk_abc123..."
```

**Example Response:**

```json
{
  "symbol": "TSLA",
  "models": {
    "random_forest": {
      "signal": "BUY",
      "confidence": 0.72,
      "predicted_change": 5.3
    },
    "lstm": {
      "signal": "BUY",
      "confidence": 0.68,
      "predicted_price": 245.50
    },
    "arima": {
      "signal": "HOLD",
      "confidence": 0.55,
      "forecast": [240.2, 242.1, 243.8]
    }
  },
  "consensus": {
    "signal": "BUY",
    "confidence": 0.65,
    "agreement": 0.67
  }
}
```

**Python Example:**

```python
response = requests.get(
    f"{BASE_URL}/predictions/TSLA",
    headers=headers
)

predictions = response.json()
print(f"Consensus Signal: {predictions['consensus']['signal']}")

for model, pred in predictions['models'].items():
    print(f"{model}: {pred['signal']} ({pred['confidence']:.1%})")
```

### 4. Get Risk Assessment

**Endpoint:** `GET /api/v1/risk/{symbol}`

**Description:** Get comprehensive risk analysis

**Example Request:**

```bash
curl -X GET "https://api.stockaipro.com/v1/risk/NVDA" \
  -H "Authorization: Bearer sk_abc123..."
```

**Example Response:**

```json
{
  "symbol": "NVDA",
  "risk_level": "Moderate",
  "risk_score": 45,
  "volatility": 0.38,
  "max_drawdown": 0.22,
  "beta": 1.35,
  "sharpe_ratio": 1.82,
  "factors": [
    "Moderate volatility (38%)",
    "Beta above 1 indicates market sensitivity"
  ],
  "recommendation": "Appropriate for balanced portfolios with moderate risk tolerance"
}
```

### 5. Get Portfolio Data

**Endpoint:** `GET /api/v1/portfolio`

**Description:** Retrieve user's paper trading portfolio

**Example Request:**

```bash
curl -X GET "https://api.stockaipro.com/v1/portfolio" \
  -H "Authorization: Bearer sk_abc123..."
```

**Example Response:**

```json
{
  "cash_balance": 85420.50,
  "total_invested": 14579.50,
  "total_value": 16234.75,
  "total_pnl": 1655.25,
  "pnl_percentage": 11.36,
  "holdings": [
    {
      "symbol": "AAPL",
      "quantity": 50,
      "avg_price": 165.30,
      "current_price": 178.45,
      "invested": 8265.00,
      "current_value": 8922.50,
      "pnl": 657.50,
      "pnl_percentage": 7.95
    }
  ]
}
```

### 6. Execute Trade

**Endpoint:** `POST /api/v1/portfolio/trade`

**Description:** Execute a paper trade

**Example Request:**

```bash
curl -X POST "https://api.stockaipro.com/v1/portfolio/trade" \
  -H "Authorization: Bearer sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 10,
    "price": 178.45
  }'
```

**Example Response:**

```json
{
  "success": true,
  "message": "Order executed successfully",
  "trade": {
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 10,
    "price": 178.45,
    "total": 1784.50,
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "new_balance": 83636.00
}
```

### 7. Get Watchlist

**Endpoint:** `GET /api/v1/watchlist`

**Description:** Get user's watchlist

**Example Request:**

```bash
curl -X GET "https://api.stockaipro.com/v1/watchlist" \
  -H "Authorization: Bearer sk_abc123..."
```

**Example Response:**

```json
{
  "watchlist": [
    {
      "symbol": "AAPL",
      "current_price": 178.45,
      "change_percent": 1.34,
      "alert_price": 180.00,
      "added_date": "2024-01-01T00:00:00Z"
    },
    {
      "symbol": "MSFT",
      "current_price": 385.20,
      "change_percent": 0.85,
      "alert_price": null,
      "added_date": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### 8. Get Market Sentiment

**Endpoint:** `GET /api/v1/sentiment/{symbol}`

**Description:** Get real-time market sentiment analysis

**Example Request:**

```bash
curl -X GET "https://api.stockaipro.com/v1/sentiment/TSLA" \
  -H "Authorization: Bearer sk_abc123..."
```

**Example Response:**

```json
{
  "symbol": "TSLA",
  "sentiment": "Positive",
  "score": 0.68,
  "sources_analyzed": 45,
  "confidence": 0.82,
  "breakdown": {
    "news": 0.72,
    "social_media": 0.64,
    "analyst_ratings": 0.75
  },
  "trending_topics": [
    "Q4 earnings beat",
    "Production targets",
    "New model launch"
  ],
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 9. Chat with AI Assistant

**Endpoint:** `POST /api/v1/ai/chat`

**Description:** Ask questions to the AI assistant

**Example Request:**

```bash
curl -X POST "https://api.stockaipro.com/v1/ai/chat" \
  -H "Authorization: Bearer sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is RSI and how do I use it?",
    "context": {
      "symbol": "AAPL"
    }
  }'
```

**Example Response:**

```json
{
  "response": "RSI (Relative Strength Index) measures momentum...",
  "related_resources": [
    "https://docs.stockaipro.com/indicators/rsi",
    "https://docs.stockaipro.com/tutorials/technical-analysis"
  ],
  "suggested_actions": [
    "View RSI for AAPL",
    "Set RSI alerts"
  ]
}
```

### 10. Get Portfolio Optimization

**Endpoint:** `POST /api/v1/portfolio/optimize`

**Description:** Get AI-powered portfolio allocation recommendations

**Example Request:**

```bash
curl -X POST "https://api.stockaipro.com/v1/portfolio/optimize" \
  -H "Authorization: Bearer sk_abc123..." \
  -H "Content-Type: application/json" \
  -d '{
    "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "risk_tolerance": "moderate",
    "investment_amount": 10000
  }'
```

**Example Response:**

```json
{
  "allocations": {
    "AAPL": {
      "weight": 0.28,
      "amount": 2800,
      "shares": 15
    },
    "MSFT": {
      "weight": 0.31,
      "amount": 3100,
      "shares": 8
    },
    "GOOGL": {
      "weight": 0.23,
      "amount": 2300,
      "shares": 16
    },
    "AMZN": {
      "weight": 0.18,
      "amount": 1800,
      "shares": 12
    }
  },
  "expected_return": 0.12,
  "expected_risk": 0.18,
  "sharpe_ratio": 0.67,
  "diversification_score": 0.85
}
```

## 📝 Rate Limits

| Tier | Requests/Day | Requests/Minute |
|------|--------------|-----------------|
| Free | 10 | 1 |
| Basic | 100 | 10 |
| Pro | 1,000 | 100 |
| Enterprise | Unlimited | Unlimited |

## ❌ Error Handling

**Example Error Response:**

```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Stock symbol 'XYZ123' not found",
    "details": "Please verify the stock symbol and try again"
  }
}
```

**Common Error Codes:**

- `INVALID_API_KEY` - API key is invalid or expired
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INVALID_SYMBOL` - Stock symbol not found
- `INSUFFICIENT_PERMISSIONS` - Feature not available in your plan
- `VALIDATION_ERROR` - Invalid request parameters
- `SERVER_ERROR` - Internal server error

## 🔄 Webhooks (Enterprise)

Configure webhooks to receive real-time notifications:

**Available Events:**
- `price.alert` - Price alert triggered
- `portfolio.update` - Portfolio value changed
- `prediction.generated` - New ML prediction available
- `news.published` - Relevant news for watchlist stocks

**Webhook Payload Example:**

```json
{
  "event": "price.alert",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "symbol": "AAPL",
    "alert_price": 180.00,
    "current_price": 180.50,
    "change": 0.50
  }
}
```

## 📚 SDK Libraries

### Python SDK

```bash
pip install stockai-sdk
```

```python
from stockai import StockAI

client = StockAI(api_key="sk_abc123...")

# Get stock data
stock = client.stocks.get("AAPL")
print(f"Price: ${stock.current_price}")

# Get AI analysis
analysis = client.ai.analyze("AAPL")
print(analysis.summary)

# Execute trade
trade = client.portfolio.buy("AAPL", quantity=10)
print(f"Trade executed: {trade.status}")
```

### JavaScript/Node.js SDK

```bash
npm install @stockai/sdk
```

```javascript
const StockAI = require('@stockai/sdk');

const client = new StockAI('sk_abc123...');

// Get stock data
const stock = await client.stocks.get('AAPL');
console.log(`Price: $${stock.current_price}`);

// Get predictions
const predictions = await client.predictions.get('AAPL');
console.log(`Signal: ${predictions.consensus.signal}`);
```

## 🔗 Additional Resources

- [Full API Reference](https://docs.stockaipro.com/api)
- [API Status Page](https://status.stockaipro.com)
- [API Changelog](https://docs.stockaipro.com/changelog)
- [Postman Collection](https://www.postman.com/stockaipro)

## 💬 Support

Need help with the API?

- 📧 Email: api-support@stockaipro.com
- 💬 Discord: [Join our community](https://discord.gg/stockaipro)
- 📚 Documentation: [docs.stockaipro.com](https://docs.stockaipro.com)
