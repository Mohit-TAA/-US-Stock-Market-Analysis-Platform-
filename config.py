"""
Configuration Module for US Stock Market Analysis Platform - SaaS Version
"""
import os
from pathlib import Path
from typing import Dict

class Config:
    APP_NAME = "StockAI Pro"
    VERSION = "2.0.0"
    APP_URL = os.getenv("APP_URL", "http://localhost:8501")
    
    DB_PATH = os.getenv("DATABASE_URL", "data/pro_us_market.db")
    CACHE_DIR = "data/cache"
    EXPORT_DIR = "data/exports"
    REPORT_DIR = "data/reports"
    DEFAULT_BALANCE = 100000.0
    COMMISSION_RATE = 0.0
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
    SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "noreply@stockaipro.com")
    
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret-change-in-production")
    
    MAJOR_INDICES: Dict[str, str] = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI', 
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT',
        'S&P 400 MidCap': '^MID',
        'NYSE Composite': '^NYA',
        'CBOE Volatility': '^VIX',
        'S&P 100': '^OEX',
        'NASDAQ 100': '^NDX',
        'Dow Jones Utilities': '^DJU',
        'Dow Jones Transportation': '^DJT',
        'Wilshire 5000': '^W5000'
    }
    
    SUBSCRIPTION_TIERS = {
        'free': {
            'name': 'Free',
            'price': 0,
            'stock_limit': 5,
            'features': [
                'Basic stock analysis',
                'Technical indicators',
                'Paper trading',
                'Market dashboard'
            ],
            'ai_features': False,
            'api_access': False,
            'priority_support': False
        },
        'basic': {
            'name': 'Basic',
            'price': 9.99,
            'stock_limit': 50,
            'features': [
                'All Free features',
                'Advanced charting',
                'ML predictions',
                'Basic AI insights',
                'Email alerts'
            ],
            'ai_features': 'basic',
            'api_access': False,
            'priority_support': False
        },
        'pro': {
            'name': 'Pro',
            'price': 29.99,
            'stock_limit': -1,
            'features': [
                'All Basic features',
                'Unlimited stocks',
                'Full AI insights',
                'AI chatbot',
                'Sentiment analysis',
                'Deep learning predictions',
                'Portfolio optimization',
                'Advanced forecasting'
            ],
            'ai_features': 'full',
            'api_access': True,
            'priority_support': False
        },
        'enterprise': {
            'name': 'Enterprise',
            'price': 99.99,
            'stock_limit': -1,
            'features': [
                'All Pro features',
                'White-label option',
                'Priority support',
                'Custom integrations',
                'Dedicated account manager',
                'SLA guarantee'
            ],
            'ai_features': 'full',
            'api_access': True,
            'priority_support': True
        }
    }

for directory in [Config.CACHE_DIR, Config.EXPORT_DIR, Config.REPORT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
    'JPM', 'JNJ', 'V', 'WMT', 'DIS', 'NFLX', 'AMD', 'INTC',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'BAC', 'XOM', 'PFE',
    'CSCO', 'ORCL', 'IBM', 'CRM', 'PYPL', 'ADBE', 'COIN'
]
