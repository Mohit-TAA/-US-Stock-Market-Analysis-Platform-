# StockAI Pro - SaaS Setup Guide

## 🚀 Overview

This guide will help you deploy StockAI Pro as a fully functional SaaS platform with AI features, user authentication, and subscription management.

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL or MySQL (recommended for production)
- Stripe account (for payments)
- OpenAI or Anthropic API key (for AI features)
- SendGrid account (for emails)
- News API / Finnhub API keys (for sentiment analysis)

## 🔧 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/stockai

# Application
APP_URL=https://yourdomain.com
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# AI Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Payment (Stripe)
STRIPE_PUBLISHABLE_KEY=pk_...
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Email (SendGrid)
SENDGRID_API_KEY=SG...
SENDGRID_FROM_EMAIL=noreply@yourdomain.com

# Market Data APIs
NEWS_API_KEY=...
FINNHUB_API_KEY=...
```

### 3. Initialize Database

```bash
python -c "from auth import AuthManager; AuthManager()"
```

This will create the necessary tables for user authentication and management.

## 🎯 Features

### Core Features (All Tiers)
- Real-time market data
- Technical analysis
- Paper trading
- Market dashboard

### AI Features (Pro/Enterprise)
- AI-powered stock summaries
- Risk assessment
- Sentiment analysis
- Natural language chatbot
- Portfolio optimization

### SaaS Infrastructure
- User authentication (email/password)
- Subscription management
- API access (Pro/Enterprise)
- Usage tracking
- Audit logging

## 💳 Subscription Tiers

### Free
- 5 stocks analysis
- Basic features
- Paper trading

### Basic ($9.99/month)
- 50 stocks
- Basic AI insights
- Email alerts

### Pro ($29.99/month)
- Unlimited stocks
- Full AI features
- API access
- Advanced forecasting

### Enterprise ($99.99/month)
- All Pro features
- Priority support
- White-label option
- Custom integrations

## 🚀 Deployment

### Option 1: Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add environment variables in Settings
4. Deploy

### Option 2: Docker

```bash
docker build -t stockai-pro .
docker run -p 8501:8501 --env-file .env stockai-pro
```

### Option 3: Custom Server

```bash
streamlit run app_saas.py --server.port 8501 --server.address 0.0.0.0
```

## 🔐 Authentication Flow

1. User registers with email/password
2. Email verification (optional)
3. Login creates session token
4. Session persists for 7 days
5. API key generated automatically

## 💰 Payment Integration

### Stripe Setup

1. Create products in Stripe Dashboard:
   - Basic Plan: $9.99/month
   - Pro Plan: $29.99/month
   - Enterprise Plan: $99.99/month

2. Configure webhook endpoint:
   - URL: `https://yourdomain.com/webhook/stripe`
   - Events: `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`

3. Test with Stripe CLI:
   ```bash
   stripe listen --forward-to localhost:8501/webhook/stripe
   ```

## 🤖 AI Configuration

### OpenAI Integration

```python
# Automatically used if OPENAI_API_KEY is set
# Models: gpt-3.5-turbo for cost efficiency
```

### Anthropic Integration

```python
# Fallback if OpenAI not available
# Models: claude-3-sonnet-20240229
```

### Fallback Mode

If no AI API keys are configured, the system uses rule-based analysis.

## 📊 API Access

### Endpoint Examples

```bash
# Get stock data
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.stockaipro.com/v1/stocks/AAPL

# Get AI analysis
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.stockaipro.com/v1/analysis/AAPL

# Get predictions
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.stockaipro.com/v1/predictions/AAPL
```

## 🔍 Monitoring & Analytics

### User Metrics
- Total users
- Active subscriptions
- Churn rate
- Feature usage

### Technical Metrics
- API response times
- Error rates
- Database performance
- Cache hit rates

## 🛡️ Security

### Best Practices

1. **Password Security**
   - PBKDF2 hashing with SHA-256
   - Unique salt per user
   - 100,000 iterations

2. **Session Management**
   - 7-day session expiry
   - Secure token generation
   - Session invalidation on logout

3. **API Security**
   - Rate limiting (coming soon)
   - API key rotation
   - Request validation

4. **Data Privacy**
   - Encrypted database connections
   - GDPR compliance
   - Data retention policies

## 📧 Email Notifications

### Types of Emails
- Welcome email
- Email verification
- Password reset
- Subscription confirmation
- Payment receipts
- Price alerts
- Portfolio updates

### Configuration

```python
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Configure in config.py
SENDGRID_API_KEY = "your-key"
SENDGRID_FROM_EMAIL = "noreply@yourdomain.com"
```

## 🎨 Customization

### Branding

Edit `config.py`:

```python
APP_NAME = "Your Brand Name"
LOGO_URL = "https://yourdomain.com/logo.png"
PRIMARY_COLOR = "#1f77b4"
```

### Feature Toggles

```python
FEATURES = {
    'ai_insights': True,
    'paper_trading': True,
    'api_access': True,
    'white_label': False
}
```

## 🐛 Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Check DATABASE_URL
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL
```

**Stripe Webhook Not Working**
```bash
# Verify webhook secret
stripe webhooks list

# Check webhook logs
stripe webhooks logs
```

**AI API Rate Limits**
```bash
# Monitor usage
# OpenAI: https://platform.openai.com/usage
# Anthropic: https://console.anthropic.com/usage
```

## 📈 Scaling

### Performance Optimization

1. **Caching**
   - Redis for session storage
   - Cache market data
   - CDN for static assets

2. **Database**
   - Connection pooling
   - Read replicas
   - Query optimization

3. **Load Balancing**
   - Multiple Streamlit instances
   - Nginx reverse proxy
   - Auto-scaling

## 🔄 Updates & Maintenance

### Version Updates

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations
python migrate.py

# Restart application
systemctl restart stockai
```

### Database Backups

```bash
# PostgreSQL backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql

# Restore
psql $DATABASE_URL < backup_YYYYMMDD.sql
```

## 📞 Support

- **Documentation:** https://docs.stockaipro.com
- **Email:** support@stockaipro.com
- **GitHub Issues:** https://github.com/yourusername/stockai-pro/issues

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

---

**Built with ❤️ using Streamlit, Python, and AI**
