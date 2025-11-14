# 🎓 StockAI Pro - Complete Tutorial

## Welcome! 👋

This tutorial will walk you through every feature of your new AI-powered stock analysis SaaS platform.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Running the Application](#running-the-application)
3. [User Features](#user-features)
4. [AI Features](#ai-features)
5. [API Usage](#api-usage)
6. [Customization](#customization)
7. [Deployment](#deployment)

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /home/engine/project

# Install required packages
pip install -r requirements.txt
```

### Step 2: Configure Environment (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.
```

**For testing, you can skip this step!** The app works without API keys using fallback systems.

### Step 3: Initialize Database

```bash
# This creates the user database automatically
python -c "from auth import AuthManager; AuthManager()"
```

---

## Running the Application

### Option 1: Original Freemium Version

```bash
streamlit run US_Market_Analysis_Platform_Freemium_version.py
```

**Features:**
- 5 free stock analyses
- All technical analysis tools
- Paper trading
- No user accounts needed

### Option 2: New SaaS Version

```bash
streamlit run app_saas.py
```

**Features:**
- User authentication
- Multiple subscription tiers
- AI-powered insights
- API access (Pro+)
- Portfolio optimization

---

## User Features

### 1. Registration & Login

#### Creating an Account

1. Click **"Login / Register"** in sidebar
2. Switch to **"Register"** tab
3. Fill in details:
   - Email: `test@example.com`
   - Full Name: `Test User`
   - Password: `password123` (min 8 chars)
4. Check **"I agree to terms"**
5. Click **"Create Account"**

#### Logging In

1. Switch to **"Login"** tab
2. Enter email and password
3. Click **"Login"**
4. Redirected to dashboard

### 2. Market Dashboard

**Access:** Available to all users

**Features:**
- Major US indices (S&P 500, Dow Jones, NASDAQ)
- Real-time price updates
- Quick stock lookup
- Market overview

**Try it:**
```
1. Go to "Market Dashboard"
2. View current index performance
3. Enter stock symbol (e.g., AAPL)
4. Click "Analyze"
```

### 3. Stock Analysis

**Access:** Free tier = 5 stocks, Paid = More/Unlimited

**What you get:**
- Current price and change
- Volume data
- 52-week range
- AI-powered summary (if Pro+)
- Technical indicators
- Interactive charts

**Example Analysis:**

```
Stock: AAPL
Symbol: Enter "AAPL"

You'll see:
- Current Price: $178.45
- Change: +1.34%
- Volume: 52.8M
- AI Summary: "Apple shows strong bullish momentum..."
```

### 4. Advanced Charts

**Access:** All tiers

**Features:**
- Price & Volume charts
- Technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Moving Averages (SMA/EMA)
- Multiple timeframes
- Zoom and pan

**How to use:**
```
1. Go to "Advanced Charts"
2. Select timeframe (1mo, 3mo, 6mo, 1y, 2y)
3. View multiple indicator panels
4. Hover for detailed values
```

### 5. AI Predictions

**Access:** Pro+ only

**What you get:**
- Multiple ML model predictions
- Buy/Sell/Hold signals
- Confidence scores
- Consensus recommendation

**Models included:**
- Random Forest Classifier
- Technical Analysis Composite
- Momentum Trend Predictor
- Volume-based Predictor

**Example output:**
```
Random Forest Model
├─ Signal: BUY
├─ Confidence: 72%
└─ Features: RSI, MACD, Volume

Technical Analysis
├─ Signal: BUY
├─ Confidence: 65%
└─ Indicators: 8/12 bullish

Consensus: STRONG BUY (85% confidence)
```

### 6. Advanced Forecasting

**Access:** Pro+ only

**Models:**
- Linear Regression (trend-based)
- AR(1) Model (autoregressive)
- Monte Carlo Simulation (probabilistic)
- ARIMA (time series)

**How to use:**
```
1. Go to "Forecasting"
2. Select stock symbol
3. Choose forecast model
4. Set prediction horizon (days)
5. View projected prices with confidence intervals
```

### 7. Paper Trading

**Access:** All tiers

**Features:**
- $100,000 starting virtual balance
- Buy/sell stocks at current prices
- Track portfolio performance
- View profit/loss
- No real money risk

**Getting started:**
```
1. Go to "Paper Trading"
2. View your portfolio summary
3. Enter stock symbol to trade
4. Place buy order:
   - Quantity: 10 shares
   - Price: Market price
5. Track in holdings table
6. Sell when ready
```

**Example portfolio:**
```
Cash Balance: $85,420
Invested: $14,580
Current Value: $16,235
Total P&L: +$1,655 (+11.4%)

Holdings:
AAPL: 50 shares @ $165.30 → Now $178.45 (+7.9%)
MSFT: 20 shares @ $350.00 → Now $385.20 (+10.1%)
```

### 8. AI Assistant

**Access:** Pro+ only

**What it does:**
- Answer stock questions
- Explain technical indicators
- Platform navigation help
- Investment education

**Example conversations:**

**You:** "What is RSI?"

**Bot:**
```
RSI (Relative Strength Index) measures momentum:
- Above 70: Overbought (potential reversal)
- Below 30: Oversold (potential reversal)
- 50-70: Bullish momentum
- 30-50: Bearish momentum

Best used with other indicators!
```

**You:** "Should I buy AAPL?"

**Bot:**
```
I can't give specific advice, but consider:
1. Your financial goals
2. Risk tolerance
3. Diversification
4. Company fundamentals
5. Current market conditions

Use our analysis tools to research thoroughly!
```

### 9. Professional Reports

**Access:** All tiers (limited features in Free)

**Report types:**
- PDF Report (comprehensive analysis)
- HTML Report (interactive)
- CSV Export (raw data)

**What's included:**
- Stock overview
- Technical analysis
- Fundamental data
- AI insights (Pro+)
- ML predictions (Pro+)
- Charts and graphs

---

## AI Features

### 1. AI Stock Summary

**How it works:**
1. Analyzes price data
2. Reviews fundamentals
3. Considers ML predictions
4. Generates natural language summary

**Example:**

Input: AAPL stock data

Output:
```
AAPL Analysis Summary

Overall Assessment: BULLISH

Apple Inc. is currently trading at $178.45, showing 
strong upward momentum with a 12.3% gain over the 
analysis period.

Technical Indicators:
- RSI: Neutral (58.2)
- MACD: Bullish crossover detected
- Volume: Above average

Key Strengths:
- Strong revenue growth (15% YoY)
- Solid profit margins (25%)
- Positive technical momentum

Investment Recommendation: BUY
The combination of strong fundamentals and positive 
technical signals suggests favorable conditions.

*Automated analysis. Always conduct thorough research.*
```

### 2. Risk Assessment

**Metrics calculated:**
- Volatility (annualized)
- Maximum drawdown
- RSI extremes
- Volume volatility
- Risk score (0-100)

**Example output:**
```
Risk Assessment: NVDA

Risk Level: Moderate
Risk Score: 45/100

Metrics:
- Volatility: 38% (annualized)
- Max Drawdown: 22%
- Beta: 1.35 (more volatile than market)

Risk Factors:
- Moderate volatility (38%)
- Beta above 1 indicates market sensitivity
- Occasional volume spikes

Recommendation:
Appropriate for balanced portfolios with moderate 
risk tolerance. Consider position sizing carefully.
```

### 3. Sentiment Analysis

**Data sources:**
- News headlines
- Financial reports
- Social media mentions
- Analyst ratings

**Sentiment scale:**
- Very Positive: +0.7 to +1.0
- Positive: +0.2 to +0.7
- Neutral: -0.2 to +0.2
- Negative: -0.7 to -0.2
- Very Negative: -1.0 to -0.7

**Example:**
```
Sentiment Analysis: TSLA

Overall Sentiment: Positive
Score: +0.68/1.0
Confidence: 82%

Sources Analyzed: 45
- News: +0.72
- Social Media: +0.64
- Analyst Ratings: +0.75

Trending Topics:
- Q4 earnings beat expectations
- Production targets met
- New model launch announced
```

### 4. Portfolio Optimization

**How it works:**
1. Enter stock symbols
2. Specify risk tolerance
3. AI calculates optimal allocation
4. Provides rebalancing recommendations

**Example:**

Input:
```
Stocks: AAPL, MSFT, GOOGL, AMZN
Risk Tolerance: Moderate
Investment: $10,000
```

Output:
```
Optimized Portfolio

Allocations:
- AAPL: 28% ($2,800) - 15 shares
- MSFT: 31% ($3,100) - 8 shares
- GOOGL: 23% ($2,300) - 16 shares
- AMZN: 18% ($1,800) - 12 shares

Metrics:
- Expected Return: 12%
- Expected Risk: 18%
- Sharpe Ratio: 0.67
- Diversification Score: 85%

Recommendation:
Well-diversified portfolio balanced for moderate 
risk. Rebalance quarterly to maintain allocations.
```

---

## API Usage

### Getting Your API Key

1. Upgrade to Pro or Enterprise
2. Go to Account Settings
3. Copy your API key
4. Keep it secure!

### Authentication

Include in every request:
```bash
Authorization: Bearer YOUR_API_KEY
```

### Example: Get Stock Data

```bash
curl -X GET "https://api.stockaipro.com/v1/stocks/AAPL" \
  -H "Authorization: Bearer sk_abc123..."
```

### Example: Python SDK

```python
from stockai import StockAI

client = StockAI(api_key="your-api-key")

# Get stock data
stock = client.stocks.get("AAPL")
print(f"Price: ${stock.current_price}")

# Get AI analysis
analysis = client.ai.analyze("AAPL")
print(analysis.summary)

# Get predictions
predictions = client.predictions.get("AAPL")
print(f"Signal: {predictions.consensus.signal}")
```

For complete API documentation, see `API_EXAMPLES.md`.

---

## Customization

### Change Branding

Edit `config.py`:
```python
class Config:
    APP_NAME = "Your Company Name"
    APP_URL = "https://yourdomain.com"
```

### Adjust Pricing

Edit `config.py`:
```python
SUBSCRIPTION_TIERS = {
    'basic': {
        'name': 'Basic',
        'price': 19.99,  # Change here
        'stock_limit': 100,  # Change limit
        # ...
    }
}
```

### Toggle Features

```python
# Enable/disable features
ENABLE_AI_FEATURES = True
ENABLE_PAPER_TRADING = True
ENABLE_API_ACCESS = True
```

### Add New Indicators

Edit original platform file, add to `_calculate_advanced_indicators`:
```python
def _calculate_advanced_indicators(self, df):
    # Existing indicators...
    
    # Add your custom indicator
    df['my_indicator'] = calculate_my_indicator(df)
    
    return df
```

---

## Deployment

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Select `app_saas.py`
5. Add secrets in settings:
   ```toml
   OPENAI_API_KEY = "sk-..."
   STRIPE_SECRET_KEY = "sk_..."
   # etc.
   ```
6. Deploy!

### Option 2: Docker

```bash
# Build image
docker build -t stockai-pro .

# Run container
docker run -p 8501:8501 --env-file .env stockai-pro

# Or use docker-compose
docker-compose up -d
```

### Option 3: AWS/GCP/Azure

1. Set up compute instance
2. Install dependencies
3. Configure environment
4. Set up reverse proxy (Nginx)
5. Enable SSL/HTTPS
6. Configure auto-scaling

For detailed instructions, see `SAAS_SETUP_GUIDE.md`.

---

## Tips & Best Practices

### For Users

1. **Start with Free Tier** - Test before upgrading
2. **Use Paper Trading** - Practice without risk
3. **Combine Signals** - Don't rely on single indicator
4. **Diversify** - Don't put all eggs in one basket
5. **Set Alerts** - Monitor your stocks
6. **Review Reports** - Export for record-keeping

### For Developers

1. **Monitor Logs** - Watch for errors
2. **Track Metrics** - User engagement, performance
3. **Test Thoroughly** - Before production deploy
4. **Backup Database** - Automated backups
5. **Update Dependencies** - Security patches
6. **Optimize Queries** - Database performance
7. **Cache Data** - Reduce API calls

### For Operators

1. **Monitor Uptime** - Use status page
2. **Review Costs** - API usage, hosting
3. **Support Users** - Quick response times
4. **Gather Feedback** - Continuous improvement
5. **Market Effectively** - Content, social media
6. **A/B Testing** - Optimize conversions
7. **Security Audits** - Regular checks

---

## Troubleshooting

### App won't start

```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Clear Streamlit cache
streamlit cache clear
```

### Database errors

```bash
# Reset database
rm data/users.db
python -c "from auth import AuthManager; AuthManager()"
```

### AI features not working

Check `.env` file:
```bash
# Verify API keys are set
cat .env | grep API_KEY

# Test OpenAI connection
python -c "from ai_engine import AIInsightsEngine; print(AIInsightsEngine().provider)"
```

### Payment errors

1. Check Stripe keys in `.env`
2. Verify webhook endpoint configured
3. Test with Stripe test cards
4. Review Stripe dashboard logs

---

## Next Steps

### For Learning

1. Complete this tutorial ✅
2. Try all features
3. Read AI Features Guide
4. Review API Examples
5. Study code structure

### For Testing

1. Run locally
2. Create test accounts
3. Test all subscription tiers
4. Try API endpoints
5. Generate reports

### For Production

1. Read Launch Checklist
2. Set up production services
3. Configure monitoring
4. Prepare marketing
5. Launch! 🚀

---

## Resources

### Documentation
- [Quick Start](QUICK_START.md)
- [Setup Guide](SAAS_SETUP_GUIDE.md)
- [AI Features](AI_FEATURES_GUIDE.md)
- [API Examples](API_EXAMPLES.md)
- [Launch Checklist](LAUNCH_CHECKLIST.md)

### External Links
- [Streamlit Docs](https://docs.streamlit.io)
- [OpenAI API](https://platform.openai.com/docs)
- [Stripe Docs](https://stripe.com/docs)
- [Yahoo Finance](https://finance.yahoo.com)

### Community
- Discord: [Join community]
- GitHub: [Contribute]
- Twitter: [@stockaipro]

---

## Congratulations! 🎉

You've completed the tutorial! You now know how to:

✅ Install and run the application  
✅ Use all features  
✅ Understand AI capabilities  
✅ Access the API  
✅ Customize the platform  
✅ Deploy to production  

**Ready to launch your SaaS?** Follow the Launch Checklist!

**Questions?** Check the documentation or join our community!

**Good luck!** 🚀
