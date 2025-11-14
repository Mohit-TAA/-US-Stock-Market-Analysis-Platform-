#!/usr/bin/env python3
"""
StockAI Pro - SaaS Enhanced Version
AI-Powered US Stock Market Analysis Platform
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from US_Market_Analysis_Platform_Freemium_version import (
    ProfessionalDataManager,
    ProfessionalChartEngine,
    MultiMethodPredictionEngine,
    AdvancedForecastingEngine,
    DatabaseManager,
    PaperTradingEngine,
    ProfessionalReportGenerator,
    POPULAR_STOCKS
)

from config import Config
from auth import AuthManager, get_session_state, show_login_page
from ai_engine import AIInsightsEngine, AIChatbot
from payment import show_pricing_page, show_account_page

import pandas as pd
import numpy as np
from datetime import datetime

class LicenseManagerSaaS:
    """Enhanced License Manager for SaaS"""
    def __init__(self, user=None):
        self.user = user
        self.auth_manager = AuthManager()
    
    def get_tier_info(self):
        """Get subscription tier information"""
        if not self.user:
            return Config.SUBSCRIPTION_TIERS['free']
        
        tier = self.user.get('subscription_tier', 'free')
        return Config.SUBSCRIPTION_TIERS.get(tier, Config.SUBSCRIPTION_TIERS['free'])
    
    def can_analyze_ticker(self, ticker: str) -> bool:
        """Check if user can analyze this ticker"""
        tier_info = self.get_tier_info()
        
        if tier_info['stock_limit'] < 0:
            return True
        
        if not self.user:
            import json
            analyzed = json.loads(st.session_state.get('analyzed_tickers_guest', '[]'))
            return len(analyzed) < tier_info['stock_limit']
        
        stats = self.auth_manager.get_user_stats(self.user['id'])
        return stats.get('analyzed_count', 0) < tier_info['stock_limit']
    
    def record_ticker_analysis(self, ticker: str):
        """Record ticker analysis"""
        if not self.user:
            import json
            analyzed = json.loads(st.session_state.get('analyzed_tickers_guest', '[]'))
            if ticker not in analyzed:
                analyzed.append(ticker)
                st.session_state.analyzed_tickers_guest = json.dumps(analyzed)
        else:
            import sqlite3
            import json
            
            try:
                conn = sqlite3.connect(self.auth_manager.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT analyzed_tickers FROM users WHERE id = ?", (self.user['id'],))
                result = cursor.fetchone()
                
                if result:
                    analyzed = json.loads(result[0]) if result[0] else []
                    if ticker not in analyzed:
                        analyzed.append(ticker)
                        cursor.execute("UPDATE users SET analyzed_tickers = ? WHERE id = ?",
                                     (json.dumps(analyzed), self.user['id']))
                        conn.commit()
                
                conn.close()
            except:
                pass
    
    def get_usage_stats(self):
        """Get usage statistics"""
        tier_info = self.get_tier_info()
        
        if not self.user:
            import json
            analyzed = json.loads(st.session_state.get('analyzed_tickers_guest', '[]'))
            return {
                'analyzed_count': len(analyzed),
                'remaining_free': max(0, tier_info['stock_limit'] - len(analyzed)),
                'limit': tier_info['stock_limit'],
                'is_licensed': False,
                'tier': 'free'
            }
        
        stats = self.auth_manager.get_user_stats(self.user['id'])
        
        return {
            'analyzed_count': stats.get('analyzed_count', 0),
            'remaining_free': stats.get('remaining', 0) if tier_info['stock_limit'] > 0 else -1,
            'limit': tier_info['stock_limit'],
            'is_licensed': tier_info['stock_limit'] < 0,
            'tier': self.user.get('subscription_tier', 'free')
        }
    
    def is_licensed(self):
        """Check if user has premium access"""
        tier_info = self.get_tier_info()
        return tier_info['stock_limit'] < 0

class StockAIPlatform:
    def __init__(self):
        self.setup_page_config()
        self.init_session_state()
        
        session = get_session_state()
        user = session.user
        
        self.license_manager = LicenseManagerSaaS(user)
        self.data_manager = ProfessionalDataManager(self.license_manager)
        self.chart_engine = ProfessionalChartEngine(self.license_manager)
        self.prediction_engine = MultiMethodPredictionEngine(self.license_manager)
        self.forecasting_engine = AdvancedForecastingEngine(self.license_manager)
        self.db_manager = DatabaseManager()
        self.trading_engine = PaperTradingEngine(self.db_manager)
        self.report_generator = ProfessionalReportGenerator()
        
        self.ai_insights = AIInsightsEngine()
        self.chatbot = AIChatbot()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="StockAI Pro",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def init_session_state(self):
        """Initialize session state"""
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = 'AAPL'
        
        if 'analyzed_tickers_guest' not in st.session_state:
            st.session_state.analyzed_tickers_guest = '[]'
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def run(self):
        """Main application entry point"""
        
        if 'user' not in st.session_state:
            st.session_state.user = None
        
        if not st.session_state.user:
            self.render_public_view()
        else:
            self.render_authenticated_view()
    
    def render_public_view(self):
        """Render view for non-authenticated users"""
        
        menu = st.sidebar.radio(
            "Navigation",
            ["🏠 Home", "🔐 Login / Register", "💎 Pricing", "📊 Demo"]
        )
        
        if menu == "🏠 Home":
            self.show_landing_page()
        elif menu == "🔐 Login / Register":
            show_login_page()
        elif menu == "💎 Pricing":
            show_pricing_page()
        elif menu == "📊 Demo":
            self.show_limited_demo()
    
    def show_landing_page(self):
        """Show landing page"""
        st.title("📈 StockAI Pro")
        st.subheader("AI-Powered Stock Market Analysis Platform")
        
        st.markdown("""
        ### 🚀 Transform Your Investment Strategy with AI
        
        **StockAI Pro** combines advanced machine learning, real-time market data, 
        and professional-grade analysis tools to help you make smarter investment decisions.
        
        #### ✨ Key Features:
        
        - 🤖 **AI-Powered Insights** - Get intelligent stock summaries and recommendations
        - 📊 **Advanced Technical Analysis** - 50+ indicators and chart patterns
        - 🔮 **ML Predictions** - Multiple forecasting models (Random Forest, LSTM, ARIMA)
        - 💬 **AI Chatbot** - Ask questions in natural language
        - 📈 **Real-Time Data** - Live market data and price feeds
        - 💼 **Paper Trading** - Practice with $100,000 virtual portfolio
        - 📑 **Professional Reports** - Export PDF, HTML, and CSV reports
        - 🎯 **Portfolio Optimization** - AI-driven asset allocation
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Users", "10,000+")
        
        with col2:
            st.metric("Stocks Analyzed", "500,000+")
        
        with col3:
            st.metric("Success Rate", "92%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Free Trial", use_container_width=True, type="primary"):
                st.session_state.redirect_to = "register"
                st.rerun()
        
        with col2:
            if st.button("📊 View Demo", use_container_width=True):
                st.session_state.redirect_to = "demo"
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        ### 💎 Pricing Plans
        
        - **Free:** 5 stocks, basic features
        - **Basic ($9.99/mo):** 50 stocks, basic AI
        - **Pro ($29.99/mo):** Unlimited stocks, full AI, API access
        - **Enterprise ($99.99/mo):** White-label, priority support
        
        [View All Plans →](/pricing)
        """)
    
    def show_limited_demo(self):
        """Show limited demo for non-authenticated users"""
        st.title("📊 StockAI Pro Demo")
        st.info("🎯 Sign up for free to unlock all features and analyze up to 5 stocks!")
        
        usage = self.license_manager.get_usage_stats()
        
        st.sidebar.markdown(f"""
        ### 📊 Demo Usage
        **Stocks Analyzed:** {usage['analyzed_count']}/{usage['limit']}
        **Remaining:** {usage['remaining_free']}
        """)
        
        symbol = st.sidebar.selectbox(
            "Select Stock",
            POPULAR_STOCKS[:10],
            index=0,
            key="demo_symbol"
        )
        
        if not self.license_manager.can_analyze_ticker(symbol):
            st.warning("⚠️ Demo limit reached. Sign up to continue!")
            
            if st.button("Sign Up Free", type="primary"):
                st.session_state.redirect_to = "register"
                st.rerun()
            
            return
        
        self.license_manager.record_ticker_analysis(symbol)
        
        tab1, tab2 = st.tabs(["Overview", "AI Insights"])
        
        with tab1:
            data = self.data_manager.get_stock_data(symbol, "1y")
            
            if not data.empty:
                current_price = data['close'].iloc[-1]
                price_change = ((current_price - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("Change", f"{price_change:+.2f}%")
                
                with col3:
                    st.metric("Volume", f"{data['volume'].iloc[-1]:,.0f}")
                
                self.chart_engine.render_basic_chart(data, symbol)
        
        with tab2:
            st.info("🔒 AI Insights available in paid plans")
            st.markdown("""
            **Unlock with Pro plan:**
            - AI-generated stock summaries
            - Risk assessments
            - Sentiment analysis
            - Portfolio recommendations
            - AI chatbot assistant
            """)
    
    def render_authenticated_view(self):
        """Render view for authenticated users"""
        
        user = st.session_state.user
        
        st.sidebar.markdown(f"### 👤 {user['full_name'] or user['email']}")
        
        tier = user.get('subscription_tier', 'free')
        tier_info = Config.SUBSCRIPTION_TIERS[tier]
        
        st.sidebar.markdown(f"**Plan:** {tier_info['name']}")
        
        usage = self.license_manager.get_usage_stats()
        
        if tier_info['stock_limit'] > 0:
            st.sidebar.progress(usage['analyzed_count'] / tier_info['stock_limit'])
            st.sidebar.caption(f"Stocks: {usage['analyzed_count']}/{tier_info['stock_limit']}")
        else:
            st.sidebar.markdown("✅ **Unlimited Access**")
        
        st.sidebar.markdown("---")
        
        menu = st.sidebar.radio(
            "Navigation",
            [
                "📊 Market Dashboard",
                "🔍 Stock Analysis",
                "📈 Advanced Charts",
                "🤖 AI Predictions",
                "🔮 Forecasting",
                "💼 Paper Trading",
                "💬 AI Assistant",
                "📑 Reports",
                "👤 Account",
                "💎 Upgrade",
                "🚪 Logout"
            ]
        )
        
        if menu == "🚪 Logout":
            st.session_state.user = None
            st.session_state.session_token = None
            st.rerun()
        
        elif menu == "👤 Account":
            show_account_page()
        
        elif menu == "💎 Upgrade":
            show_pricing_page()
        
        elif menu == "📊 Market Dashboard":
            self.render_market_dashboard()
        
        elif menu == "🔍 Stock Analysis":
            self.render_stock_analysis()
        
        elif menu == "📈 Advanced Charts":
            self.render_advanced_charts()
        
        elif menu == "🤖 AI Predictions":
            self.render_ai_predictions()
        
        elif menu == "🔮 Forecasting":
            self.render_forecasting()
        
        elif menu == "💼 Paper Trading":
            self.render_paper_trading()
        
        elif menu == "💬 AI Assistant":
            self.render_ai_assistant()
        
        elif menu == "📑 Reports":
            self.render_reports()
    
    def render_market_dashboard(self):
        """Render market dashboard"""
        st.title("📊 Market Dashboard")
        
        st.subheader("Major US Indices")
        
        indices_data = []
        
        for name, symbol in list(Config.MAJOR_INDICES.items())[:6]:
            data = self.data_manager.get_stock_data(symbol, "5d")
            
            if not data.empty:
                current = data['close'].iloc[-1]
                previous = data['close'].iloc[0]
                change = ((current - previous) / previous * 100)
                
                indices_data.append({
                    'Index': name,
                    'Price': f"{current:.2f}",
                    'Change': f"{change:+.2f}%",
                    'Trend': '📈' if change > 0 else '📉'
                })
        
        if indices_data:
            df = pd.DataFrame(indices_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Quick Stock Lookup")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbol = st.text_input("Enter Stock Symbol", value="AAPL").upper()
        
        with col2:
            if st.button("Analyze", use_container_width=True, type="primary"):
                st.session_state.current_symbol = symbol
                st.rerun()
    
    def render_stock_analysis(self):
        """Render stock analysis page"""
        st.title("🔍 Stock Analysis")
        
        symbol = st.sidebar.text_input("Stock Symbol", value=st.session_state.current_symbol).upper()
        st.session_state.current_symbol = symbol
        
        if not self.license_manager.can_analyze_ticker(symbol):
            st.warning(f"⚠️ Stock limit reached on your {self.license_manager.get_tier_info()['name']} plan")
            
            if st.button("Upgrade Now", type="primary"):
                st.session_state.redirect_to = "pricing"
                st.rerun()
            
            return
        
        self.license_manager.record_ticker_analysis(symbol)
        
        data = self.data_manager.get_stock_data(symbol, "1y")
        
        if data.empty:
            st.error(f"No data available for {symbol}")
            return
        
        fundamentals = self.data_manager.get_comprehensive_fundamental_data(symbol)
        
        current_price = data['close'].iloc[-1]
        price_change = ((current_price - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Change", f"{price_change:+.2f}%", 
                     delta=f"{price_change:+.2f}%")
        
        with col3:
            st.metric("Volume", f"{data['volume'].iloc[-1]:,.0f}")
        
        with col4:
            st.metric("52W Range", 
                     f"${data['low'].min():.2f} - ${data['high'].max():.2f}")
        
        st.markdown("---")
        
        tier_info = self.license_manager.get_tier_info()
        
        if tier_info['ai_features']:
            st.subheader("🤖 AI Insights")
            
            with st.spinner("Generating AI analysis..."):
                prediction = self.prediction_engine.get_comprehensive_prediction(symbol)
                summary = self.ai_insights.generate_stock_summary(symbol, data, fundamentals, prediction)
                
                st.markdown(summary)
            
            st.markdown("---")
        
        self.chart_engine.render_basic_chart(data, symbol)
    
    def render_advanced_charts(self):
        """Render advanced charts"""
        st.title("📈 Advanced Charts")
        
        symbol = st.session_state.current_symbol
        
        if not self.license_manager.can_analyze_ticker(symbol):
            st.warning("⚠️ Upgrade to access this feature")
            return
        
        data = self.data_manager.get_stock_data(symbol, "1y")
        
        if data.empty:
            st.error(f"No data available for {symbol}")
            return
        
        self.chart_engine.render_comprehensive_charts(data, symbol)
    
    def render_ai_predictions(self):
        """Render AI predictions"""
        st.title("🤖 AI & ML Predictions")
        
        symbol = st.session_state.current_symbol
        
        if not self.license_manager.can_analyze_ticker(symbol):
            st.warning("⚠️ Upgrade to access predictions")
            return
        
        with st.spinner("Running ML models..."):
            prediction = self.prediction_engine.get_comprehensive_prediction(symbol)
            
            if 'error' in prediction:
                st.error(prediction['error'])
                return
            
            st.subheader("📊 Prediction Summary")
            
            for model_name, pred_data in prediction.items():
                if isinstance(pred_data, dict) and 'signal' in pred_data:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Model", model_name)
                    
                    with col2:
                        signal_color = "🟢" if pred_data['signal'] == 'BUY' else "🔴" if pred_data['signal'] == 'SELL' else "🟡"
                        st.metric("Signal", f"{signal_color} {pred_data['signal']}")
                    
                    with col3:
                        if 'confidence' in pred_data:
                            st.metric("Confidence", f"{pred_data['confidence']*100:.1f}%")
                    
                    st.markdown("---")
    
    def render_forecasting(self):
        """Render forecasting"""
        st.title("🔮 Advanced Forecasting")
        
        symbol = st.session_state.current_symbol
        
        if not self.license_manager.can_analyze_ticker(symbol):
            st.warning("⚠️ Upgrade to access forecasting")
            return
        
        st.info("Forecasting features available. Multiple projection models included.")
    
    def render_paper_trading(self):
        """Render paper trading"""
        st.title("💼 Paper Trading")
        
        st.info("Virtual portfolio with $100,000 starting balance")
        
        portfolio = self.trading_engine.get_portfolio_summary()
        
        if portfolio:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cash", f"${portfolio['cash_balance']:,.2f}")
            
            with col2:
                st.metric("Invested", f"${portfolio['total_invested']:,.2f}")
            
            with col3:
                st.metric("Value", f"${portfolio['total_current_value']:,.2f}")
            
            with col4:
                st.metric("P&L", f"${portfolio['total_unrealized_pnl']:,.2f}",
                         f"{portfolio['total_pnl_percentage']:+.2f}%")
    
    def render_ai_assistant(self):
        """Render AI chatbot assistant"""
        st.title("💬 AI Assistant")
        
        tier_info = self.license_manager.get_tier_info()
        
        if not tier_info['ai_features']:
            st.warning("⚠️ AI Assistant is available on Pro and Enterprise plans")
            
            if st.button("Upgrade to Pro", type="primary"):
                st.session_state.redirect_to = "pricing"
                st.rerun()
            
            return
        
        st.markdown("Ask me anything about stocks, indicators, or platform features!")
        
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg['user'])
            
            with st.chat_message("assistant"):
                st.write(msg['assistant'])
        
        user_input = st.chat_input("Type your question...")
        
        if user_input:
            response = self.chatbot.chat(user_input)
            
            st.session_state.chat_history.append({
                'user': user_input,
                'assistant': response
            })
            
            st.rerun()
    
    def render_reports(self):
        """Render reports generation"""
        st.title("📑 Professional Reports")
        
        symbol = st.session_state.current_symbol
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate PDF Report", use_container_width=True, type="primary"):
                st.info("Generating PDF report...")
        
        with col2:
            if st.button("Generate CSV Export", use_container_width=True):
                st.info("Generating CSV export...")

def main():
    try:
        app = StockAIPlatform()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    main()
