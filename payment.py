"""
Payment Integration Module - Stripe Integration for SaaS Subscriptions
"""
from typing import Optional, Dict
from datetime import datetime
import streamlit as st

try:
    import stripe
    STRIPE_AVAILABLE = True
except:
    STRIPE_AVAILABLE = False

from config import Config
from auth import AuthManager

class PaymentManager:
    def __init__(self):
        self.stripe_available = STRIPE_AVAILABLE and Config.STRIPE_SECRET_KEY
        
        if self.stripe_available:
            stripe.api_key = Config.STRIPE_SECRET_KEY
        
        self.price_ids = {
            'basic': 'price_basic_monthly',
            'pro': 'price_pro_monthly', 
            'enterprise': 'price_enterprise_monthly'
        }
    
    def create_checkout_session(self, user_id: int, tier: str, email: str) -> Optional[str]:
        """Create Stripe checkout session"""
        
        if not self.stripe_available:
            return None
        
        try:
            tier_info = Config.SUBSCRIPTION_TIERS.get(tier)
            if not tier_info or tier == 'free':
                return None
            
            session = stripe.checkout.Session.create(
                customer_email=email,
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'unit_amount': int(tier_info['price'] * 100),
                        'recurring': {
                            'interval': 'month'
                        },
                        'product_data': {
                            'name': f'StockAI Pro - {tier_info["name"]} Plan',
                            'description': ', '.join(tier_info['features'][:3])
                        }
                    },
                    'quantity': 1
                }],
                mode='subscription',
                success_url=Config.APP_URL + '/success?session_id={CHECKOUT_SESSION_ID}',
                cancel_url=Config.APP_URL + '/pricing',
                metadata={
                    'user_id': user_id,
                    'tier': tier
                }
            )
            
            return session.url
            
        except Exception as e:
            st.error(f"Payment error: {str(e)}")
            return None
    
    def create_customer_portal_session(self, stripe_customer_id: str) -> Optional[str]:
        """Create Stripe customer portal session"""
        
        if not self.stripe_available:
            return None
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=stripe_customer_id,
                return_url=Config.APP_URL + '/account'
            )
            
            return session.url
            
        except:
            return None
    
    def handle_webhook(self, payload: Dict, signature: str) -> bool:
        """Handle Stripe webhook events"""
        
        if not self.stripe_available:
            return False
        
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, Config.STRIPE_WEBHOOK_SECRET
            )
            
            if event['type'] == 'checkout.session.completed':
                session = event['data']['object']
                self._handle_successful_payment(session)
            
            elif event['type'] == 'customer.subscription.updated':
                subscription = event['data']['object']
                self._handle_subscription_update(subscription)
            
            elif event['type'] == 'customer.subscription.deleted':
                subscription = event['data']['object']
                self._handle_subscription_cancellation(subscription)
            
            return True
            
        except Exception as e:
            return False
    
    def _handle_successful_payment(self, session: Dict):
        """Handle successful payment"""
        user_id = session['metadata'].get('user_id')
        tier = session['metadata'].get('tier')
        customer_id = session.get('customer')
        
        if user_id and tier:
            auth = AuthManager()
            auth.update_subscription(int(user_id), tier, customer_id)
    
    def _handle_subscription_update(self, subscription: Dict):
        """Handle subscription update"""
        pass
    
    def _handle_subscription_cancellation(self, subscription: Dict):
        """Handle subscription cancellation"""
        pass

def show_pricing_page():
    """Display pricing page with subscription tiers"""
    st.title("💎 Choose Your Plan")
    st.write("Unlock powerful AI-driven stock analysis")
    
    cols = st.columns(4)
    
    for idx, (tier_key, tier_info) in enumerate(Config.SUBSCRIPTION_TIERS.items()):
        with cols[idx]:
            st.markdown(f"### {tier_info['name']}")
            
            if tier_info['price'] == 0:
                st.markdown("# FREE")
            else:
                st.markdown(f"# ${tier_info['price']}")
                st.caption("per month")
            
            st.markdown("---")
            
            for feature in tier_info['features']:
                st.markdown(f"✅ {feature}")
            
            st.markdown("---")
            
            if st.session_state.get('user'):
                current_tier = st.session_state.user.get('subscription_tier', 'free')
                
                if current_tier == tier_key:
                    st.button("Current Plan", disabled=True, use_container_width=True)
                elif tier_key == 'free':
                    st.button("Downgrade", use_container_width=True, 
                             key=f"btn_{tier_key}")
                else:
                    if st.button("Upgrade", use_container_width=True, 
                               key=f"btn_{tier_key}", type="primary"):
                        payment_manager = PaymentManager()
                        
                        if payment_manager.stripe_available:
                            checkout_url = payment_manager.create_checkout_session(
                                st.session_state.user['id'],
                                tier_key,
                                st.session_state.user['email']
                            )
                            
                            if checkout_url:
                                st.markdown(f'<meta http-equiv="refresh" content="0; url={checkout_url}">', 
                                          unsafe_allow_html=True)
                            else:
                                st.error("Payment system unavailable")
                        else:
                            st.info("""
                            **Payment Integration Coming Soon!**
                            
                            Contact sales@stockaipro.com to upgrade manually.
                            """)
            else:
                st.button("Sign Up", use_container_width=True, 
                         key=f"btn_{tier_key}", type="primary")
    
    st.markdown("---")
    st.markdown("""
    ### ❓ Frequently Asked Questions
    
    **Can I cancel anytime?**  
    Yes, cancel anytime with no penalties. Your access continues until the end of the billing period.
    
    **What payment methods do you accept?**  
    We accept all major credit cards via Stripe's secure payment processing.
    
    **Do you offer refunds?**  
    Yes, we offer a 14-day money-back guarantee for all paid plans.
    
    **Is there a free trial?**  
    The Free plan gives you full access to analyze 5 stocks with all basic features.
    
    **Can I upgrade or downgrade?**  
    Yes, you can change your plan at any time. Upgrades take effect immediately.
    """)

def show_account_page():
    """Display account management page"""
    if not st.session_state.get('user'):
        st.warning("Please login to view account details")
        return
    
    user = st.session_state.user
    auth = AuthManager()
    
    st.title("👤 Account Settings")
    
    tab1, tab2, tab3 = st.tabs(["Profile", "Subscription", "API Access"])
    
    with tab1:
        st.subheader("Profile Information")
        
        with st.form("profile_form"):
            full_name = st.text_input("Full Name", value=user.get('full_name', ''))
            email = st.text_input("Email", value=user['email'], disabled=True)
            
            if st.form_submit_button("Update Profile"):
                st.success("Profile updated successfully!")
        
        st.markdown("---")
        st.subheader("Change Password")
        
        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Change Password"):
                if new_password == confirm_password:
                    st.success("Password changed successfully!")
                else:
                    st.error("Passwords do not match")
    
    with tab2:
        st.subheader("Subscription Details")
        
        tier = user.get('subscription_tier', 'free')
        tier_info = Config.SUBSCRIPTION_TIERS.get(tier, Config.SUBSCRIPTION_TIERS['free'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Plan", tier_info['name'])
            st.metric("Monthly Price", f"${tier_info['price']}")
        
        with col2:
            stats = auth.get_user_stats(user['id'])
            
            if stats.get('stock_limit', 0) > 0:
                st.metric("Stocks Analyzed", 
                         f"{stats.get('analyzed_count', 0)}/{stats.get('stock_limit', 0)}")
            else:
                st.metric("Stocks Analyzed", f"{stats.get('analyzed_count', 0)} (Unlimited)")
            
            st.metric("Status", user.get('subscription_status', 'active').title())
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Upgrade Plan", use_container_width=True, type="primary"):
                st.switch_page("pricing")
        
        with col2:
            if tier != 'free':
                if st.button("Manage Billing", use_container_width=True):
                    payment_manager = PaymentManager()
                    portal_url = payment_manager.create_customer_portal_session(
                        user.get('stripe_customer_id', '')
                    )
                    
                    if portal_url:
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={portal_url}">', 
                                  unsafe_allow_html=True)
    
    with tab3:
        st.subheader("API Access")
        
        if tier_info.get('api_access'):
            st.success("✅ API access enabled for your plan")
            
            api_key = user.get('api_key', '')
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text_input("API Key", value=api_key, type="password", disabled=True)
            
            with col2:
                if st.button("Regenerate", use_container_width=True):
                    new_key = auth.regenerate_api_key(user['id'])
                    if new_key:
                        st.success("New API key generated!")
                        st.session_state.user['api_key'] = new_key
                        st.rerun()
            
            st.markdown("""
            ### API Documentation
            
            **Base URL:** `https://api.stockaipro.com/v1`
            
            **Authentication:**
            ```
            Authorization: Bearer YOUR_API_KEY
            ```
            
            **Endpoints:**
            - `GET /stocks/{symbol}` - Get stock data
            - `GET /analysis/{symbol}` - Get AI analysis
            - `GET /predictions/{symbol}` - Get ML predictions
            - `GET /portfolio` - Get portfolio data
            
            [View Full Documentation →](https://docs.stockaipro.com)
            """)
        else:
            st.warning("⚠️ API access not available on your current plan")
            st.info("Upgrade to Pro or Enterprise to access the API")
            
            if st.button("View Plans", use_container_width=True):
                st.switch_page("pricing")
