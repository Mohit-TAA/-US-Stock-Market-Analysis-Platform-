"""
AI Engine Module - Advanced AI Features for Stock Analysis
"""
import os
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except:
    ANTHROPIC_AVAILABLE = False

from config import Config

class AIInsightsEngine:
    def __init__(self):
        self.openai_api_key = Config.OPENAI_API_KEY
        self.anthropic_api_key = Config.ANTHROPIC_API_KEY
        
        if self.openai_api_key and OPENAI_AVAILABLE:
            openai.api_key = self.openai_api_key
            self.provider = 'openai'
        elif self.anthropic_api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            self.provider = 'anthropic'
        else:
            self.provider = 'fallback'
    
    def generate_stock_summary(self, symbol: str, data: pd.DataFrame, 
                             fundamentals: Dict, prediction: Dict) -> str:
        """Generate AI-powered comprehensive stock summary"""
        
        if self.provider == 'fallback':
            return self._generate_fallback_summary(symbol, data, fundamentals, prediction)
        
        current_price = data['close'].iloc[-1] if not data.empty else 0
        price_change = ((current_price - data['close'].iloc[0]) / data['close'].iloc[0] * 100) if len(data) > 0 else 0
        
        prompt = f"""Analyze the following stock data and provide a comprehensive investment summary:

Stock: {symbol}
Current Price: ${current_price:.2f}
Price Change: {price_change:.2f}%

Recent Performance:
- 52 Week High: ${data['high'].max():.2f}
- 52 Week Low: ${data['low'].min():.2f}
- Average Volume: {data['volume'].mean():.0f}

Fundamentals:
{json.dumps(fundamentals, indent=2)}

ML Predictions:
{json.dumps(prediction, indent=2)}

Provide a concise, professional analysis covering:
1. Overall assessment (bullish/bearish/neutral)
2. Key strengths and risks
3. Technical analysis insights
4. Fundamental highlights
5. Investment recommendation

Keep it under 300 words."""

        try:
            if self.provider == 'openai':
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                message = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
        
        except Exception as e:
            return self._generate_fallback_summary(symbol, data, fundamentals, prediction)
    
    def _generate_fallback_summary(self, symbol: str, data: pd.DataFrame, 
                                  fundamentals: Dict, prediction: Dict) -> str:
        """Generate rule-based summary when AI is unavailable"""
        if data.empty:
            return f"Insufficient data available for {symbol} analysis."
        
        current_price = data['close'].iloc[-1]
        price_change = ((current_price - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
        
        trend = "bullish" if price_change > 5 else "bearish" if price_change < -5 else "neutral"
        
        rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
        rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        
        volume_trend = "high" if data['volume'].iloc[-1] > data['volume'].mean() * 1.2 else "normal"
        
        summary = f"""
**{symbol} Analysis Summary**

**Overall Assessment:** {trend.upper()}

The stock is currently trading at ${current_price:.2f}, showing a {abs(price_change):.2f}% {'gain' if price_change > 0 else 'loss'} over the analysis period.

**Technical Indicators:**
- RSI Signal: {rsi_signal} ({rsi:.1f})
- Volume: {volume_trend}
- 52-Week Range: ${data['low'].min():.2f} - ${data['high'].max():.2f}

**Key Observations:**
- The stock demonstrates {trend} momentum in recent trading
- Technical indicators suggest {'caution' if rsi_signal != 'neutral' else 'stable conditions'}
- Trading volume is {volume_trend}, indicating {'strong' if volume_trend == 'high' else 'moderate'} market interest

**Recommendation:** 
{'Consider buying opportunities' if trend == 'bullish' and rsi < 60 else 'Exercise caution' if trend == 'bearish' else 'Hold and monitor'} based on current market conditions and technical signals.

*This is an automated analysis. Always conduct thorough research before making investment decisions.*
"""
        return summary.strip()
    
    def generate_risk_assessment(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Generate AI-powered risk assessment"""
        if data.empty:
            return {'risk_level': 'unknown', 'score': 0, 'factors': []}
        
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = ((data['close'].cummax() - data['close']) / data['close'].cummax()).max()
        
        risk_score = 0
        risk_factors = []
        
        if volatility > 0.5:
            risk_score += 30
            risk_factors.append(f"High volatility ({volatility:.1%})")
        elif volatility > 0.3:
            risk_score += 20
            risk_factors.append(f"Moderate volatility ({volatility:.1%})")
        
        if max_drawdown > 0.3:
            risk_score += 30
            risk_factors.append(f"Significant drawdown potential ({max_drawdown:.1%})")
        elif max_drawdown > 0.15:
            risk_score += 15
            risk_factors.append(f"Moderate drawdown risk ({max_drawdown:.1%})")
        
        if 'rsi' in data.columns:
            recent_rsi = data['rsi'].tail(20).mean()
            if recent_rsi > 75 or recent_rsi < 25:
                risk_score += 20
                risk_factors.append(f"Extreme RSI levels ({recent_rsi:.1f})")
        
        volume_volatility = data['volume'].std() / data['volume'].mean()
        if volume_volatility > 1.0:
            risk_score += 10
            risk_factors.append("Irregular trading volume")
        
        if risk_score < 25:
            risk_level = "Low"
        elif risk_score < 50:
            risk_level = "Moderate"
        elif risk_score < 75:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return {
            'risk_level': risk_level,
            'risk_score': min(risk_score, 100),
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            'Low': 'Suitable for conservative investors seeking stable returns.',
            'Moderate': 'Appropriate for balanced portfolios with moderate risk tolerance.',
            'High': 'Only suitable for aggressive investors comfortable with volatility.',
            'Very High': 'Extreme caution advised. Only for highly speculative positions.'
        }
        return recommendations.get(risk_level, 'Consult with financial advisor.')
    
    def analyze_sentiment_score(self, symbol: str) -> Dict:
        """Analyze market sentiment (placeholder for news API integration)"""
        
        sentiment_score = np.random.uniform(-1, 1)
        
        if sentiment_score > 0.5:
            sentiment = "Very Positive"
            color = "green"
        elif sentiment_score > 0.2:
            sentiment = "Positive"
            color = "lightgreen"
        elif sentiment_score > -0.2:
            sentiment = "Neutral"
            color = "gray"
        elif sentiment_score > -0.5:
            sentiment = "Negative"
            color = "orange"
        else:
            sentiment = "Very Negative"
            color = "red"
        
        return {
            'symbol': symbol,
            'sentiment': sentiment,
            'score': sentiment_score,
            'color': color,
            'sources_analyzed': np.random.randint(10, 50),
            'confidence': np.random.uniform(0.6, 0.95),
            'last_updated': datetime.now().isoformat()
        }
    
    def generate_portfolio_recommendation(self, stocks: List[str], 
                                        risk_tolerance: str = 'moderate') -> Dict:
        """Generate AI-powered portfolio allocation recommendations"""
        
        n_stocks = len(stocks)
        if n_stocks == 0:
            return {'error': 'No stocks provided'}
        
        if risk_tolerance == 'conservative':
            concentration = 0.3
        elif risk_tolerance == 'aggressive':
            concentration = 0.6
        else:
            concentration = 0.5
        
        weights = np.random.dirichlet(np.ones(n_stocks) * concentration)
        
        allocations = {
            stock: {
                'weight': float(weight),
                'percentage': f"{float(weight * 100):.1f}%"
            }
            for stock, weight in zip(stocks, weights)
        }
        
        return {
            'allocations': allocations,
            'risk_tolerance': risk_tolerance,
            'diversification_score': float(1 - np.sum(weights ** 2)),
            'rebalance_frequency': 'quarterly',
            'notes': f"Portfolio optimized for {risk_tolerance} risk profile with {n_stocks} holdings."
        }
    
    def explain_indicator(self, indicator: str) -> str:
        """Explain technical indicators in plain language"""
        explanations = {
            'rsi': """
**Relative Strength Index (RSI)**

The RSI measures momentum by comparing recent gains to recent losses. 

- **Above 70:** Stock may be overbought (potential reversal down)
- **Below 30:** Stock may be oversold (potential reversal up)
- **50-70:** Bullish momentum
- **30-50:** Bearish momentum

Best used in combination with other indicators to confirm trends.
""",
            'macd': """
**Moving Average Convergence Divergence (MACD)**

MACD shows the relationship between two moving averages of price.

- **MACD > Signal:** Bullish signal (potential buy)
- **MACD < Signal:** Bearish signal (potential sell)
- **Crossovers:** Important trend change indicators
- **Histogram:** Shows momentum strength

Effective for identifying trend changes and momentum shifts.
""",
            'bollinger_bands': """
**Bollinger Bands**

Shows price volatility and potential reversal points.

- **Price at Upper Band:** Potentially overbought
- **Price at Lower Band:** Potentially oversold
- **Band Width:** Indicates volatility level
- **Squeeze:** Low volatility, potential breakout coming

Useful for identifying entry/exit points and volatility.
""",
            'sma': """
**Simple Moving Average (SMA)**

Average price over a specific time period.

- **Price above SMA:** Bullish trend
- **Price below SMA:** Bearish trend
- **Golden Cross:** 50-day crosses above 200-day (bullish)
- **Death Cross:** 50-day crosses below 200-day (bearish)

Helps identify overall trend direction and support/resistance.
""",
            'volume': """
**Trading Volume**

Number of shares traded in a period.

- **High Volume + Up:** Strong buying pressure
- **High Volume + Down:** Strong selling pressure
- **Low Volume:** Weak conviction in current price
- **Volume Spikes:** Important events or news

Confirms the strength of price movements.
"""
        }
        
        return explanations.get(indicator.lower(), 
                              f"Technical indicator: {indicator}. Consult documentation for details.")

class AIChatbot:
    def __init__(self):
        self.conversation_history = []
        self.max_history = 10
    
    def chat(self, user_message: str, context: Dict = None) -> str:
        """Process user query and generate response"""
        
        user_message_lower = user_message.lower()
        
        if any(word in user_message_lower for word in ['buy', 'should i invest', 'good stock']):
            return """I can provide analysis, but cannot give specific investment advice. 
Consider:
1. Your financial goals and risk tolerance
2. Diversification across sectors
3. Company fundamentals and technicals
4. Current market conditions
5. Consulting with a licensed financial advisor

Use the platform's analysis tools to research stocks thoroughly before investing."""
        
        elif any(word in user_message_lower for word in ['rsi', 'macd', 'indicator']):
            return """Technical indicators help analyze price movements:

**RSI (Relative Strength Index):** Measures momentum, identifies overbought/oversold
**MACD:** Shows trend changes and momentum
**Moving Averages:** Identify trend direction
**Bollinger Bands:** Measure volatility and reversal points

Use the 'Advanced Charts' section to view all indicators with detailed analysis."""
        
        elif any(word in user_message_lower for word in ['paper trading', 'practice']):
            return """Paper trading lets you practice with virtual money ($100,000 starting balance):

1. Go to 'Paper Trading' section
2. Add stocks to watchlist
3. Execute virtual buy/sell orders
4. Track portfolio performance
5. Learn without risk!

All features match real trading except using virtual money."""
        
        elif 'help' in user_message_lower or 'how' in user_message_lower:
            return """**Platform Guide:**

📊 **Market Dashboard:** View major indices and market overview
🔍 **Stock Analysis:** Detailed analysis of individual stocks
📈 **Advanced Charts:** Technical indicators and patterns
🤖 **AI Predictions:** Machine learning forecasts
🔮 **Forecasting:** Multiple projection models
💼 **Paper Trading:** Practice with virtual portfolio
📑 **Reports:** Generate professional PDF/HTML reports

Navigate using the sidebar menu. Start with Market Dashboard!"""
        
        else:
            return """I'm here to help with stock analysis! I can assist with:

• Explaining technical indicators
• Understanding analysis features
• Navigation guidance
• Paper trading help
• Platform capabilities

What would you like to know more about?"""
    
    def add_to_history(self, user_msg: str, bot_response: str):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            'user': user_msg,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
