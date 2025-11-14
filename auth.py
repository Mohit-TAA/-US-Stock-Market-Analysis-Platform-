"""
Authentication and User Management Module for SaaS Platform
"""
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import streamlit as st
from config import Config

class AuthManager:
    def __init__(self, db_path: str = "data/users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize user authentication database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                subscription_tier TEXT DEFAULT 'free',
                subscription_status TEXT DEFAULT 'active',
                subscription_start_date TEXT,
                subscription_end_date TEXT,
                analyzed_tickers TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                email_verified INTEGER DEFAULT 0,
                api_key TEXT UNIQUE,
                stripe_customer_id TEXT,
                reset_token TEXT,
                reset_token_expiry TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                request_count INTEGER DEFAULT 0,
                date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, pwd_hash = password_hash.split('$')
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return new_hash.hex() == pwd_hash
        except:
            return False
    
    def register_user(self, email: str, password: str, full_name: str = "") -> Tuple[bool, str]:
        """Register a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                conn.close()
                return False, "Email already registered"
            
            password_hash = self.hash_password(password)
            api_key = self.generate_api_key()
            
            cursor.execute("""
                INSERT INTO users (email, password_hash, full_name, api_key, subscription_start_date)
                VALUES (?, ?, ?, ?, ?)
            """, (email, password_hash, full_name, api_key, datetime.now().isoformat()))
            
            conn.commit()
            user_id = cursor.lastrowid
            
            self.log_action(user_id, "user_registered", f"New user registered: {email}")
            
            conn.close()
            return True, "Registration successful"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, email: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """Authenticate user and create session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, email, password_hash, full_name, subscription_tier, 
                       subscription_status, api_key
                FROM users WHERE email = ?
            """, (email,))
            
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return False, None
            
            user_id, email, password_hash, full_name, tier, status, api_key = user
            
            if not self.verify_password(password, password_hash):
                conn.close()
                return False, None
            
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE id = ?
            """, (datetime.now().isoformat(), user_id))
            
            session_token = secrets.token_urlsafe(32)
            expires_at = (datetime.now() + timedelta(days=7)).isoformat()
            
            cursor.execute("""
                INSERT INTO sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            """, (user_id, session_token, expires_at))
            
            conn.commit()
            
            self.log_action(user_id, "user_login", f"User logged in: {email}")
            
            conn.close()
            
            user_data = {
                'id': user_id,
                'email': email,
                'full_name': full_name,
                'subscription_tier': tier,
                'subscription_status': status,
                'session_token': session_token,
                'api_key': api_key
            }
            
            return True, user_data
            
        except Exception as e:
            return False, None
    
    def logout_user(self, session_token: str):
        """Logout user and invalidate session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
            conn.commit()
            conn.close()
            
        except:
            pass
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT u.id, u.email, u.full_name, u.subscription_tier, 
                       u.subscription_status, s.expires_at, u.api_key
                FROM users u
                JOIN sessions s ON u.id = s.user_id
                WHERE s.session_token = ?
            """, (session_token,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
            
            user_id, email, full_name, tier, status, expires_at, api_key = result
            
            if datetime.fromisoformat(expires_at) < datetime.now():
                return None
            
            return {
                'id': user_id,
                'email': email,
                'full_name': full_name,
                'subscription_tier': tier,
                'subscription_status': status,
                'api_key': api_key
            }
            
        except:
            return None
    
    def generate_api_key(self) -> str:
        """Generate unique API key"""
        return f"sk_{secrets.token_urlsafe(32)}"
    
    def regenerate_api_key(self, user_id: int) -> Optional[str]:
        """Regenerate API key for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            new_api_key = self.generate_api_key()
            cursor.execute("UPDATE users SET api_key = ? WHERE id = ?", (new_api_key, user_id))
            
            conn.commit()
            conn.close()
            
            self.log_action(user_id, "api_key_regenerated", "User regenerated API key")
            
            return new_api_key
            
        except:
            return None
    
    def update_subscription(self, user_id: int, tier: str, stripe_customer_id: str = None):
        """Update user subscription tier"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            end_date = None
            if tier != 'free':
                end_date = (datetime.now() + timedelta(days=30)).isoformat()
            
            cursor.execute("""
                UPDATE users 
                SET subscription_tier = ?, 
                    subscription_status = 'active',
                    subscription_end_date = ?,
                    stripe_customer_id = ?
                WHERE id = ?
            """, (tier, end_date, stripe_customer_id, user_id))
            
            conn.commit()
            conn.close()
            
            self.log_action(user_id, "subscription_updated", f"Subscription updated to: {tier}")
            
        except Exception as e:
            pass
    
    def log_action(self, user_id: int, action: str, details: str = ""):
        """Log user action to audit log"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_log (user_id, action, details)
                VALUES (?, ?, ?)
            """, (user_id, action, details))
            
            conn.commit()
            conn.close()
            
        except:
            pass
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics and usage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT subscription_tier, subscription_status, 
                       analyzed_tickers, created_at, last_login
                FROM users WHERE id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return {}
            
            tier, status, tickers_json, created_at, last_login = result
            
            import json
            analyzed_tickers = json.loads(tickers_json) if tickers_json else []
            
            from config import Config
            tier_info = Config.SUBSCRIPTION_TIERS.get(tier, Config.SUBSCRIPTION_TIERS['free'])
            
            return {
                'subscription_tier': tier,
                'subscription_status': status,
                'analyzed_count': len(analyzed_tickers),
                'stock_limit': tier_info['stock_limit'],
                'remaining': tier_info['stock_limit'] - len(analyzed_tickers) if tier_info['stock_limit'] > 0 else -1,
                'created_at': created_at,
                'last_login': last_login
            }
            
        except:
            return {}

def get_session_state():
    """Get or initialize session state for authentication"""
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None
    
    return st.session_state

def require_auth():
    """Decorator to require authentication"""
    session = get_session_state()
    
    if not session.user:
        st.warning("⚠️ Please login to access this feature")
        return False
    
    return True

def show_login_page():
    """Display login/register page"""
    st.title("🔐 Login to StockAI Pro")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                auth = st.session_state.auth_manager
                success, user_data = auth.login_user(email, password)
                
                if success:
                    st.session_state.user = user_data
                    st.session_state.session_token = user_data['session_token']
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password")
    
    with tab2:
        with st.form("register_form"):
            email = st.text_input("Email")
            full_name = st.text_input("Full Name")
            password = st.text_input("Password", type="password")
            password_confirm = st.text_input("Confirm Password", type="password")
            
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                if not agree_terms:
                    st.error("❌ Please agree to the terms and conditions")
                elif password != password_confirm:
                    st.error("❌ Passwords do not match")
                elif len(password) < 8:
                    st.error("❌ Password must be at least 8 characters")
                else:
                    auth = st.session_state.auth_manager
                    success, message = auth.register_user(email, password, full_name)
                    
                    if success:
                        st.success("✅ Registration successful! Please login.")
                    else:
                        st.error(f"❌ {message}")
