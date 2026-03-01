"""
BoothPulse AI - Geo-Intelligent Governance Intelligence Engine
Decision-Support Governance Command Center
Advanced booth-level sentiment analysis with SQL-powered analytics
Version: 4.0.0 - SQLite Edition
"""

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import httpx
import sqlite3
import os

# ==================== DATABASE CONFIGURATION ====================

DATABASE_PATH = "database.db"
SCHEMA_PATH = "schema.sql"

def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with row factory"""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_database():
    """Initialize database and run schema"""
    print("🗄️ Initializing SQLite database...")
    
    conn = get_db_connection()
    
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, 'r') as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        print("✅ Database schema initialized!")
    else:
        print("⚠️ schema.sql not found, creating tables manually...")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT CHECK(sentiment IN ('positive', 'negative', 'neutral')),
                sentiment_score REAL,
                issue TEXT,
                country TEXT DEFAULT 'India',
                state TEXT,
                city TEXT,
                latitude REAL,
                longitude REAL,
                confidence REAL DEFAULT 0.5,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                detail TEXT,
                severity TEXT CHECK(severity IN ('high', 'medium', 'low')) DEFAULT 'medium',
                alert_type TEXT,
                acknowledged INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_feedback_sentiment ON feedback(sentiment);
            CREATE INDEX IF NOT EXISTS idx_feedback_city ON feedback(city);
            CREATE INDEX IF NOT EXISTS idx_feedback_issue ON feedback(issue);
            CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
        """)
        conn.commit()
        print("✅ Tables created!")
    
    cursor = conn.execute("SELECT COUNT(*) FROM feedback")
    count = cursor.fetchone()[0]
    print(f"📊 Database contains {count} feedback entries")
    
    conn.close()

# ==================== INITIALIZE FASTAPI ====================

app = FastAPI(
    title="BoothPulse AI",
    description="Decision-Support Governance Command Center - SQL Edition",
    version="4.0.0"
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

print("🚀 Loading VADER sentiment analysis model...")
sentiment_analyzer = SentimentIntensityAnalyzer()
print("✅ Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    init_database()

# ==================== GEO DATA ====================

COUNTRIES = {"India": "IN", "USA": "US", "UK": "GB", "Canada": "CA", "Australia": "AU", "Other": "OT"}

STATES_BY_COUNTRY = {
    "India": ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Chandigarh", "Puducherry"],
    "USA": ["California", "Texas", "Florida", "New York", "Pennsylvania", "Illinois", "Ohio", "Georgia", "North Carolina", "Michigan", "Washington", "Arizona", "Massachusetts", "Tennessee", "Indiana", "Missouri", "Maryland", "Wisconsin"],
    "UK": ["England", "Scotland", "Wales", "Northern Ireland", "Greater London", "West Midlands", "Greater Manchester", "West Yorkshire", "South Yorkshire"],
    "Canada": ["Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick", "Newfoundland"],
    "Australia": ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia", "Tasmania", "Northern Territory", "Australian Capital Territory"],
    "Other": ["State 1", "State 2", "State 3", "State 4", "State 5"]
}

CITIES_BY_STATE = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik", "Aurangabad"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubli", "Mangaluru", "Belgaum"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Prayagraj", "Noida"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri", "Asansol"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur"],
    "Punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar", "Patiala"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala", "Hisar"],
    "Delhi": ["New Delhi", "Central Delhi", "South Delhi", "North Delhi", "East Delhi"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa"],
    "Himachal Pradesh": ["Shimla", "Manali", "Dharamshala", "Kullu"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Jorhat"],
    "Chandigarh": ["Chandigarh"],
    "Puducherry": ["Puducherry", "Karaikal"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Tirupati"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur", "Korba"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Rishikesh", "Nainital"],
    "California": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento"],
    "Texas": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"],
    "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville", "Naples"],
    "New York": ["New York City", "Buffalo", "Rochester", "Albany", "Syracuse"],
    "England": ["London", "Manchester", "Birmingham", "Liverpool", "Leeds", "Bristol"],
    "Scotland": ["Edinburgh", "Glasgow", "Aberdeen", "Dundee"],
    "Ontario": ["Toronto", "Ottawa", "Mississauga", "Hamilton", "London", "Brampton"],
    "Quebec": ["Montreal", "Quebec City", "Laval", "Gatineau", "Longueuil"],
    "British Columbia": ["Vancouver", "Victoria", "Surrey", "Burnaby", "Richmond"],
    "New South Wales": ["Sydney", "Newcastle", "Wollongong", "Central Coast"],
    "Victoria": ["Melbourne", "Geelong", "Ballarat", "Bendigo"],
    "Queensland": ["Brisbane", "Gold Coast", "Cairns", "Townsville", "Sunshine Coast"],
}

CITY_COORDINATES = {
    "Mumbai": {"lat": 19.0760, "lng": 72.8777}, "Delhi": {"lat": 28.6139, "lng": 77.2090},
    "New Delhi": {"lat": 28.6139, "lng": 77.2090}, "Bengaluru": {"lat": 12.9716, "lng": 77.5946},
    "Chennai": {"lat": 13.0827, "lng": 80.2707}, "Kolkata": {"lat": 22.5726, "lng": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lng": 78.4867}, "Ahmedabad": {"lat": 23.0225, "lng": 72.5714},
    "Pune": {"lat": 18.5204, "lng": 73.8567}, "Jaipur": {"lat": 26.9124, "lng": 75.7873},
    "Lucknow": {"lat": 26.8467, "lng": 80.9462}, "Surat": {"lat": 21.1702, "lng": 72.8311},
    "Bhopal": {"lat": 23.2599, "lng": 77.4126}, "Patna": {"lat": 25.5941, "lng": 85.1376},
    "Chandigarh": {"lat": 30.7333, "lng": 76.7794}, "Coimbatore": {"lat": 11.0168, "lng": 76.9558},
    "Indore": {"lat": 22.7196, "lng": 75.8577}, "Nagpur": {"lat": 21.1458, "lng": 79.0882},
    "Kochi": {"lat": 9.9312, "lng": 76.2673}, "Shimla": {"lat": 31.1048, "lng": 77.1734},
    "Guwahati": {"lat": 26.1445, "lng": 91.7362}, "Thiruvananthapuram": {"lat": 8.5241, "lng": 76.9366},
    "Ranchi": {"lat": 23.3441, "lng": 85.3096}, "Bhubaneswar": {"lat": 20.2961, "lng": 85.8245},
    "Panaji": {"lat": 15.4909, "lng": 73.8278}, "Gurugram": {"lat": 28.4595, "lng": 77.0266},
    "Noida": {"lat": 28.5355, "lng": 77.3910}, "Varanasi": {"lat": 25.3176, "lng": 82.9739},
    "New York City": {"lat": 40.7128, "lng": -74.0060}, "Los Angeles": {"lat": 34.0522, "lng": -118.2437},
    "Chicago": {"lat": 41.8781, "lng": -87.6298}, "London": {"lat": 51.5074, "lng": -0.1278},
    "Toronto": {"lat": 43.6532, "lng": -79.3832}, "Sydney": {"lat": -33.8688, "lng": 151.2093},
    "Melbourne": {"lat": -37.8136, "lng": 144.9631}, "Vancouver": {"lat": 49.2827, "lng": -123.1207},
}

ISSUE_KEYWORDS = {
    "Water": ["water", "drinking", "supply", "tank", "pipe", "drainage", "tap", "well", "borewell", "shortage", "sewage", "flood", "sanitation"],
    "Roads": ["road", "pothole", "street", "highway", "traffic", "footpath", "bridge", "construction", "pavement", "transport", "accident", "signal"],
    "Employment": ["job", "employment", "work", "salary", "unemployment", "hiring", "career", "wages", "labor", "income", "livelihood", "business"],
    "Healthcare": ["hospital", "health", "doctor", "medicine", "clinic", "medical", "treatment", "disease", "nurse", "vaccine", "ambulance", "pharmacy"],
    "Education": ["school", "education", "college", "teacher", "student", "learning", "exam", "university", "books", "scholarship", "literacy", "tuition"]
}

geo_cache: Dict[str, Dict[str, float]] = {}

# ==================== PYDANTIC MODELS ====================

class FeedbackInput(BaseModel):
    text: str
    country: str = "India"
    state: str = ""
    city: str = ""
    street: Optional[str] = None
    landmark: Optional[str] = None
    ward: Optional[str] = None
    booth_number: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

# ==================== HELPER FUNCTIONS ====================

def detect_issue_category(text: str) -> str:
    """Detect issue category using keyword matching"""
    text_lower = text.lower()
    scores = {}
    for category, keywords in ISSUE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            scores[category] = score
    if scores:
        return max(scores, key=lambda k: scores[k])
    return "General"

def map_sentiment(label: str, score: float) -> tuple:
    """Map VADER sentiment to categories"""
    if label == "POSITIVE":
        if score > 0.85:
            return "positive", score
        elif score > 0.6:
            return "positive", score * 0.9
        else:
            return "neutral", 0.5
    else:
        if score > 0.85:
            return "negative", score
        elif score > 0.6:
            return "negative", score * 0.9
        else:
            return "neutral", 0.5

async def resolve_geo_location(street: str, landmark: str, city: str, state: str, country: str) -> Optional[Dict[str, float]]:
    """Resolve coordinates using OpenStreetMap Nominatim API"""
    query_parts = [p for p in [street, landmark, city, state, country] if p]
    query = ", ".join(query_parts)
    
    if query in geo_cache:
        return geo_cache[query]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "BoothPulseAI/4.0"}
            )
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
                    geo_cache[query] = result
                    return result
    except Exception as e:
        print(f"Geo resolution error: {e}")
    return None

# ==================== SQL QUERY FUNCTIONS ====================

def get_risk_index_sql() -> int:
    """Calculate Risk Index using SQL"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
                AVG(CASE WHEN sentiment = 'negative' THEN confidence ELSE NULL END) as avg_neg_confidence
            FROM feedback
        """)
        row = cursor.fetchone()
        
        if not row or row['total'] == 0:
            return 0
        
        total = row['total']
        negative_count = row['negative_count'] or 0
        avg_confidence = row['avg_neg_confidence'] or 0.5
        
        if negative_count == 0:
            return 0
        
        negative_pct = negative_count / total
        volume_weight = min(1, total / 20)
        risk_index = int(negative_pct * avg_confidence * volume_weight * 100)
        
        return min(max(risk_index, 0), 100)

def get_sentiment_distribution_sql(time_filter: Optional[str] = None) -> Dict:
    """Get sentiment distribution using SQL"""
    time_clause = ""
    if time_filter and time_filter != "all":
        time_map = {"1h": "-1 hour", "6h": "-6 hours", "24h": "-1 day", "7d": "-7 days", "30d": "-30 days"}
        if time_filter in time_map:
            time_clause = f"WHERE created_at >= datetime('now', '{time_map[time_filter]}')"
    
    with get_db() as conn:
        cursor = conn.execute(f"""
            SELECT sentiment, COUNT(*) as count
            FROM feedback
            {time_clause}
            GROUP BY sentiment
        """)
        
        result = {"positive": 0, "negative": 0, "neutral": 0}
        for row in cursor.fetchall():
            if row['sentiment'] in result:
                result[row['sentiment']] = row['count']
        
        return result

def get_issue_distribution_sql(time_filter: Optional[str] = None) -> Dict:
    """Get issue distribution using SQL GROUP BY"""
    time_clause = ""
    if time_filter and time_filter != "all":
        time_map = {"1h": "-1 hour", "6h": "-6 hours", "24h": "-1 day", "7d": "-7 days", "30d": "-30 days"}
        if time_filter in time_map:
            time_clause = f"WHERE created_at >= datetime('now', '{time_map[time_filter]}')"
    
    with get_db() as conn:
        cursor = conn.execute(f"""
            SELECT issue, COUNT(*) as count
            FROM feedback
            {time_clause}
            GROUP BY issue
        """)
        
        result = {"Water": 0, "Roads": 0, "Employment": 0, "Healthcare": 0, "Education": 0, "General": 0}
        for row in cursor.fetchall():
            if row['issue'] in result:
                result[row['issue']] = row['count']
        
        return result

def get_city_risk_sql() -> List[Dict]:
    """Get city risk rankings using SQL aggregation"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT 
                city,
                COUNT(*) as total,
                SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
                AVG(confidence) as avg_confidence,
                ROUND(CAST(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS FLOAT) / 
                      NULLIF(COUNT(*), 0) * 100, 1) as negative_ratio
            FROM feedback
            WHERE city IS NOT NULL AND city != ''
            GROUP BY city
            ORDER BY negative DESC, total DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            total = row['total']
            negative = row['negative']
            avg_conf = row['avg_confidence'] or 0.5
            
            negative_pct = negative / total if total > 0 else 0
            volume_weight = min(1, total / 10)
            risk_score = int(negative_pct * avg_conf * volume_weight * 100)
            risk_score = min(max(risk_score, 0), 100)
            
            if risk_score <= 30:
                status = "stable"
            elif risk_score <= 60:
                status = "watchlist"
            else:
                status = "critical"
            
            results.append({
                "city": row['city'],
                "total": total,
                "positive": row['positive'],
                "negative": negative,
                "neutral": row['neutral'],
                "negative_ratio": row['negative_ratio'] or 0,
                "risk_score": risk_score,
                "status": status
            })
        
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        return results

def get_momentum_sql() -> Dict:
    """Calculate momentum using SQL - compare last 5 vs previous 5"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT sentiment FROM feedback 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        entries = cursor.fetchall()
        
        if len(entries) < 5:
            return {"change": 0, "direction": "stable", "has_data": False}
        
        def calc_score(entry_list):
            if not entry_list:
                return 0
            positive = sum(1 for e in entry_list if e['sentiment'] == 'positive')
            negative = sum(1 for e in entry_list if e['sentiment'] == 'negative')
            return (positive - negative) / len(entry_list) * 100
        
        last_5 = entries[:5]
        prev_5 = entries[5:10] if len(entries) >= 10 else entries[len(entries)//2:]
        
        last_score = calc_score(last_5)
        prev_score = calc_score(prev_5)
        
        change = last_score - prev_score
        
        if change > 5:
            direction = "improving"
        elif change < -5:
            direction = "declining"
        else:
            direction = "stable"
        
        return {
            "change": round(change, 1),
            "direction": direction,
            "has_data": True,
            "last_score": round(last_score, 1),
            "prev_score": round(prev_score, 1)
        }

def get_anomalies_sql() -> List[Dict]:
    """Detect anomalies using SQL queries"""
    anomalies = []
    
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM feedback")
        total = cursor.fetchone()['cnt']
        
        if total < 5:
            return anomalies
        
        cursor = conn.execute("""
            SELECT 
                SUM(CASE WHEN created_at >= datetime('now', '-1 hour') THEN 1 ELSE 0 END) as last_hour,
                SUM(CASE WHEN created_at >= datetime('now', '-2 hours') AND created_at < datetime('now', '-1 hour') THEN 1 ELSE 0 END) as prev_hour
            FROM feedback
        """)
        row = cursor.fetchone()
        last_hour = row['last_hour'] or 0
        prev_hour = row['prev_hour'] or 0
        
        if prev_hour > 0 and last_hour > prev_hour * 2:
            severity = "high" if last_hour > prev_hour * 3 else "medium"
            anomalies.append({
                "type": "volume_spike",
                "message": f"⚠ Volume Spike: {last_hour} entries in last hour vs {prev_hour} in previous hour",
                "severity": severity,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        cursor = conn.execute("""
            WITH recent AS (
                SELECT sentiment, ROW_NUMBER() OVER (ORDER BY created_at DESC) as rn
                FROM feedback
            )
            SELECT 
                SUM(CASE WHEN rn <= 5 AND sentiment = 'negative' THEN 1 ELSE 0 END) * 20.0 as last_5_neg_pct,
                SUM(CASE WHEN rn > 5 AND rn <= 10 AND sentiment = 'negative' THEN 1 ELSE 0 END) * 20.0 as prev_5_neg_pct
            FROM recent
            WHERE rn <= 10
        """)
        row = cursor.fetchone()
        last_neg_pct = row['last_5_neg_pct'] or 0
        prev_neg_pct = row['prev_5_neg_pct'] or 0
        
        sentiment_change = last_neg_pct - prev_neg_pct
        if sentiment_change > 25:
            severity = "high" if sentiment_change > 40 else ("medium" if sentiment_change > 30 else "low")
            anomalies.append({
                "type": "sentiment_drop",
                "message": f"⚠ Sentiment Drop: Negative sentiment increased by {round(sentiment_change, 1)}%",
                "severity": severity,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        cursor = conn.execute("""
            SELECT issue, COUNT(*) as cnt
            FROM (SELECT issue FROM feedback ORDER BY created_at DESC LIMIT 10)
            GROUP BY issue
            HAVING COUNT(*) >= 7
        """)
        for row in cursor.fetchall():
            anomalies.append({
                "type": "issue_concentration",
                "message": f"⚠ Issue Concentration: {row['issue']} dominates with {row['cnt']}/10 recent reports",
                "severity": "medium",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    return anomalies

def generate_governance_actions_sql() -> List[Dict]:
    """Generate governance actions using SQL aggregation"""
    actions = []
    
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT 
                issue,
                COUNT(*) as total,
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
                ROUND(CAST(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS FLOAT) / 
                      NULLIF(COUNT(*), 0) * 100, 1) as negative_pct
            FROM feedback
            GROUP BY issue
            HAVING total > 0
        """)
        
        action_templates = {
            "Water": {"key_issue": "Potential water supply or quality issues", "action": "Conduct immediate water supply audit and quality testing", "department": "Water Supply & Sanitation"},
            "Roads": {"key_issue": "Infrastructure degradation reported", "action": "Initiate comprehensive infrastructure review and pothole survey", "department": "Public Works & Infrastructure"},
            "Healthcare": {"key_issue": "Healthcare service quality concerns", "action": "Conduct hospital service audit and patient satisfaction survey", "department": "Health & Medical Services"},
            "Employment": {"key_issue": "Rising unemployment/livelihood issues", "action": "Launch skill development drive and job fair initiatives", "department": "Employment & Labor Welfare"},
            "Education": {"key_issue": "Educational infrastructure or quality concerns", "action": "Review school facilities and teacher allocation", "department": "Education & Training"}
        }
        
        for row in cursor.fetchall():
            issue = row['issue']
            negative_pct = row['negative_pct'] or 0
            
            if issue in action_templates and negative_pct > 40:
                urgency = "High" if negative_pct > 60 else "Medium"
                risk_level = "Critical" if negative_pct > 60 else "Elevated"
                template = action_templates[issue]
                
                actions.append({
                    "situation": f"{issue}-related complaints at {negative_pct}% negative",
                    "key_issue": template["key_issue"],
                    "risk_level": risk_level,
                    "recommended_action": template["action"],
                    "urgency": urgency,
                    "department": template["department"],
                    "negative_pct": negative_pct
                })
        
        urgency_order = {"High": 0, "Medium": 1, "Low": 2}
        actions.sort(key=lambda x: urgency_order.get(x["urgency"], 2))
    
    return actions

def get_recent_activity_sql(limit: int = 10, time_filter: Optional[str] = None) -> List[Dict]:
    """Get recent feedback activity using SQL"""
    time_clause = ""
    if time_filter and time_filter != "all":
        time_map = {"1h": "-1 hour", "6h": "-6 hours", "24h": "-1 day", "7d": "-7 days", "30d": "-30 days"}
        if time_filter in time_map:
            time_clause = f"WHERE created_at >= datetime('now', '{time_map[time_filter]}')"
    
    with get_db() as conn:
        cursor = conn.execute(f"""
            SELECT * FROM feedback
            {time_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row['id'],
                "text": row['text'],
                "sentiment": row['sentiment'],
                "issue": row['issue'],
                "city": row['city'] or "Unknown",
                "state": row['state'],
                "country": row['country'],
                "lat": row['latitude'],
                "lon": row['longitude'],
                "confidence": row['confidence'],
                "timestamp": row['created_at']
            })
        
        return results

def get_geo_data_sql(time_filter: Optional[str] = None) -> List[Dict]:
    """Get geo data for map visualization"""
    time_clause = ""
    if time_filter and time_filter != "all":
        time_map = {"1h": "-1 hour", "6h": "-6 hours", "24h": "-1 day", "7d": "-7 days", "30d": "-30 days"}
        if time_filter in time_map:
            time_clause = f"WHERE created_at >= datetime('now', '{time_map[time_filter]}')"
    
    with get_db() as conn:
        cursor = conn.execute(f"""
            SELECT latitude, longitude, sentiment, confidence, city, issue, 
                   SUBSTR(text, 1, 100) as text_preview
            FROM feedback
            {time_clause}
            ORDER BY created_at DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            if row['latitude'] and row['longitude']:
                results.append({
                    "lat": row['latitude'],
                    "lon": row['longitude'],
                    "sentiment": row['sentiment'],
                    "confidence": row['confidence'],
                    "city": row['city'] or "Unknown",
                    "issue": row['issue'],
                    "text": row['text_preview'] + "..." if len(row['text_preview'] or "") >= 100 else row['text_preview']
                })
        
        return results

def get_timeline_data_sql(limit: int = 20, time_filter: Optional[str] = None) -> List[Dict]:
    """Get timeline data for charts"""
    time_clause = ""
    if time_filter and time_filter != "all":
        time_map = {"1h": "-1 hour", "6h": "-6 hours", "24h": "-1 day", "7d": "-7 days", "30d": "-30 days"}
        if time_filter in time_map:
            time_clause = f"WHERE created_at >= datetime('now', '{time_map[time_filter]}')"
    
    with get_db() as conn:
        cursor = conn.execute(f"""
            SELECT created_at, sentiment,
                   CASE 
                       WHEN sentiment = 'positive' THEN 1
                       WHEN sentiment = 'negative' THEN -1
                       ELSE 0
                   END as score
            FROM feedback
            {time_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "timestamp": row['created_at'],
                "sentiment": row['sentiment'],
                "score": row['score']
            })
        
        return list(reversed(results))

def get_feedback_count_sql(time_filter: Optional[str] = None) -> int:
    """Get total feedback count"""
    time_clause = ""
    if time_filter and time_filter != "all":
        time_map = {"1h": "-1 hour", "6h": "-6 hours", "24h": "-1 day", "7d": "-7 days", "30d": "-30 days"}
        if time_filter in time_map:
            time_clause = f"WHERE created_at >= datetime('now', '{time_map[time_filter]}')"
    
    with get_db() as conn:
        cursor = conn.execute(f"SELECT COUNT(*) as cnt FROM feedback {time_clause}")
        return cursor.fetchone()['cnt']

def insert_feedback_sql(entry: Dict) -> Optional[int]:
    """Insert feedback into database"""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO feedback (text, sentiment, sentiment_score, issue, country, state, city, latitude, longitude, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (entry['text'], entry['sentiment'], entry.get('sentiment_score', 0), entry['issue'],
              entry['country'], entry['state'], entry['city'], entry['lat'], entry['lon'], entry['confidence']))
        return cursor.lastrowid

def insert_alert_sql(message: str, detail: str, severity: str) -> Optional[int]:
    """Insert alert into database"""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO alerts (message, detail, severity)
            VALUES (?, ?, ?)
        """, (message, detail, severity))
        return cursor.lastrowid

def get_alerts_sql() -> Dict:
    """Get alerts from database"""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 50")
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                "id": row['id'],
                "message": row['message'],
                "detail": row['detail'],
                "severity": row['severity'],
                "timestamp": row['created_at'],
                "acknowledged": bool(row['acknowledged'])
            })
        
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN acknowledged = 0 THEN 1 ELSE 0 END) as unacknowledged,
                SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_severity,
                SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium_severity,
                SUM(CASE WHEN severity = 'low' THEN 1 ELSE 0 END) as low_severity
            FROM alerts
        """)
        counts = cursor.fetchone()
        
        return {
            "alerts": alerts,
            "total": counts['total'],
            "unacknowledged": counts['unacknowledged'] or 0,
            "high_severity": counts['high_severity'] or 0,
            "medium_severity": counts['medium_severity'] or 0,
            "low_severity": counts['low_severity'] or 0
        }

def get_raw_data_sql(limit: int = 100) -> List[Dict]:
    """Get raw feedback data for table view"""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT id, text, sentiment, issue, city, state, country, confidence, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append(dict(row))
        
        return results

def generate_ai_summary_sql() -> Dict:
    """Generate AI summary using SQL aggregation"""
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) as total FROM feedback")
        total = cursor.fetchone()['total']
        
        if total == 0:
            return {
                "situation_overview": "No data available yet. Submit feedback to generate intelligence insights.",
                "dominant_issue": "N/A",
                "sentiment_trend": "N/A",
                "highest_risk_zone": "N/A",
                "recommended_action": "Begin collecting citizen feedback to enable AI analysis.",
                "confidence_level": "N/A",
                "risk_index": 0,
                "stats": {"positive_pct": 0, "negative_pct": 0, "neutral_pct": 0, "total": 0}
            }
        
        cursor = conn.execute("""
            SELECT 
                ROUND(CAST(SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) as positive_pct,
                ROUND(CAST(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) as negative_pct,
                ROUND(CAST(SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) as neutral_pct
            FROM feedback
        """)
        sent_row = cursor.fetchone()
        positive_pct = sent_row['positive_pct'] or 0
        negative_pct = sent_row['negative_pct'] or 0
        neutral_pct = sent_row['neutral_pct'] or 0
        
        if negative_pct > positive_pct and negative_pct > neutral_pct:
            dominant_sentiment = "negative"
        elif positive_pct > neutral_pct:
            dominant_sentiment = "positive"
        else:
            dominant_sentiment = "neutral"
        
        cursor = conn.execute("""
            SELECT issue, COUNT(*) as cnt FROM feedback GROUP BY issue ORDER BY cnt DESC LIMIT 1
        """)
        issue_row = cursor.fetchone()
        top_issue = issue_row['issue'] if issue_row else "General"
        top_issue_count = issue_row['cnt'] if issue_row else 0
        
        cursor = conn.execute("""
            SELECT city, ROUND(CAST(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) as neg_pct
            FROM feedback WHERE city IS NOT NULL AND city != '' GROUP BY city HAVING COUNT(*) >= 3 ORDER BY neg_pct DESC LIMIT 1
        """)
        city_row = cursor.fetchone()
        highest_risk_city = city_row['city'] if city_row else "Unknown"
        highest_risk_pct = city_row['neg_pct'] if city_row else 0
        
        momentum = get_momentum_sql()
        
        if momentum["direction"] == "improving":
            trend = f"📈 Improving (+{momentum['change']}%)"
        elif momentum["direction"] == "declining":
            trend = f"📉 Declining ({momentum['change']}%)"
        else:
            trend = "→ Stable"
        
        risk_index = get_risk_index_sql()
        
        if total >= 50:
            confidence = "High (50+ samples)"
        elif total >= 20:
            confidence = "Medium (20-49 samples)"
        else:
            confidence = "Low (<20 samples)"
        
        cursor = conn.execute("SELECT COUNT(DISTINCT city) as cnt FROM feedback WHERE city IS NOT NULL")
        city_count = cursor.fetchone()['cnt']
        
        actions = generate_governance_actions_sql()
        if actions:
            recommended = actions[0]["recommended_action"]
        elif negative_pct > 50:
            recommended = "Conduct comprehensive citizen satisfaction survey"
        elif negative_pct > 30:
            recommended = "Monitor closely and prepare intervention protocols"
        else:
            recommended = "Maintain current operations; continue monitoring"
        
        situation = f"Analyzed {total} feedback entries across {city_count} locations. "
        if dominant_sentiment == "negative":
            situation += f"⚠️ Overall sentiment is NEGATIVE ({negative_pct}%). Immediate attention recommended."
        elif dominant_sentiment == "positive":
            situation += f"✅ Overall sentiment is POSITIVE ({positive_pct}%). Public satisfaction is healthy."
        else:
            situation += f"📊 Mixed sentiment detected ({neutral_pct}% neutral). Active monitoring advised."
        
        return {
            "situation_overview": situation,
            "dominant_issue": f"{top_issue} ({top_issue_count} reports, {round(top_issue_count/total*100, 1)}% of total)",
            "sentiment_trend": trend,
            "highest_risk_zone": f"{highest_risk_city} ({round(highest_risk_pct, 1)}% negative)" if highest_risk_pct > 0 else "No high-risk zones identified",
            "recommended_action": recommended,
            "confidence_level": confidence,
            "risk_index": risk_index,
            "stats": {"positive_pct": positive_pct, "negative_pct": negative_pct, "neutral_pct": neutral_pct, "total": total}
        }

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/geo-data")
async def get_geo_data():
    """Get countries, states, and cities data"""
    return {"countries": list(COUNTRIES.keys()), "states_by_country": STATES_BY_COUNTRY, "cities_by_state": CITIES_BY_STATE}

@app.post("/analyze")
async def analyze_feedback(feedback: FeedbackInput):
    """Analyze sentiment of submitted feedback with geo-intelligence"""
    text = feedback.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Feedback text is required")
    
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        raw_label, raw_score = "POSITIVE", (compound + 1) / 2
    elif compound <= -0.05:
        raw_label, raw_score = "NEGATIVE", (-compound + 1) / 2
    else:
        raw_label, raw_score = "NEUTRAL", 0.5
    
    sentiment, confidence = map_sentiment(raw_label, raw_score)
    issue = detect_issue_category(text)
    
    lat, lon = feedback.lat, feedback.lon
    
    if (feedback.street or feedback.landmark) and (not lat or not lon):
        resolved = await resolve_geo_location(feedback.street or "", feedback.landmark or "", feedback.city, feedback.state, feedback.country)
        if resolved:
            lat, lon = resolved["lat"], resolved["lon"]
    
    if (not lat or not lon) and feedback.city in CITY_COORDINATES:
        coords = CITY_COORDINATES[feedback.city]
        lat, lon = coords["lat"], coords["lng"]
    
    if not lat or not lon:
        lat, lon = 20.5937, 78.9629
    
    entry = {"text": text, "sentiment": sentiment, "sentiment_score": compound, "issue": issue,
             "country": feedback.country, "state": feedback.state, "city": feedback.city,
             "lat": lat, "lon": lon, "confidence": round(confidence, 4)}
    
    entry_id = insert_feedback_sql(entry)
    entry["id"] = entry_id
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    alert_triggered = False
    city_risks = get_city_risk_sql()
    city_risk = next((c for c in city_risks if c["city"] == feedback.city), None)
    
    if city_risk and city_risk["status"] in ["watchlist", "critical"]:
        insert_alert_sql(
            f"🚨 Escalation Risk in {feedback.city}, {feedback.state}",
            f"Negative sentiment ratio: {city_risk['negative_ratio']}%, Risk Score: {city_risk['risk_score']}",
            "high" if city_risk["status"] == "critical" else "medium"
        )
        alert_triggered = True
    
    return {"success": True, "data": entry, "alert_triggered": alert_triggered, "message": f"Feedback analyzed: {sentiment.upper()} sentiment detected"}

@app.get("/dashboard-data")
async def get_dashboard_data(time_range: str = Query(default="all")):
    """Get comprehensive dashboard data with SQL aggregation"""
    total = get_feedback_count_sql(time_range)
    
    if total == 0:
        return {
            "total_feedback": 0, "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0, "risk_index": 0, "early_warning": False,
            "geo_data": [], "issue_distribution": {}, "timeline_data": [], "recent_activity": [],
            "city_stats": [], "momentum": {"change": 0, "direction": "stable", "has_data": False},
            "governance_actions": [], "anomalies": [], "time_range": time_range, "database_entries": 0
        }
    
    sentiment_dist = get_sentiment_distribution_sql(time_range)
    positive_pct = round((sentiment_dist["positive"] / total) * 100, 1) if total > 0 else 0
    negative_pct = round((sentiment_dist["negative"] / total) * 100, 1) if total > 0 else 0
    neutral_pct = round((sentiment_dist["neutral"] / total) * 100, 1) if total > 0 else 0
    
    risk_index = get_risk_index_sql()
    early_warning = negative_pct > 40 or risk_index > 60
    
    return {
        "total_feedback": total, "sentiment_distribution": sentiment_dist,
        "positive_pct": positive_pct, "negative_pct": negative_pct, "neutral_pct": neutral_pct,
        "risk_index": risk_index, "early_warning": early_warning,
        "geo_data": get_geo_data_sql(time_range), "issue_distribution": get_issue_distribution_sql(time_range),
        "timeline_data": get_timeline_data_sql(20, time_range), "recent_activity": get_recent_activity_sql(10, time_range),
        "city_stats": get_city_risk_sql()[:10], "momentum": get_momentum_sql(),
        "governance_actions": generate_governance_actions_sql(), "anomalies": get_anomalies_sql(),
        "time_range": time_range, "database_entries": get_feedback_count_sql()
    }

@app.get("/alerts")
async def get_alerts():
    """Get all alerts"""
    return get_alerts_sql()

@app.get("/ai-summary")
async def get_ai_summary():
    """Get AI-generated intelligence summary"""
    summary = generate_ai_summary_sql()
    
    return {
        "structured_summary": {
            "situation": summary["situation_overview"],
            "dominant_issue": summary["dominant_issue"],
            "trend": summary["sentiment_trend"].replace("📈 ", "").replace("📉 ", "").replace("→ ", ""),
            "risk_zone": summary["highest_risk_zone"],
            "action": summary["recommended_action"],
            "confidence": int(summary["stats"]["total"] / 50 * 100) if summary["stats"]["total"] < 50 else 100
        },
        "risk_index": summary["risk_index"],
        "risk_level": "Low" if summary["risk_index"] < 30 else ("Moderate" if summary["risk_index"] < 60 else "High"),
        "total_feedback": summary["stats"]["total"],
        "cities_at_risk": [c for c in get_city_risk_sql() if c["status"] in ["watchlist", "critical"]][:5],
        "stats": summary["stats"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/momentum")
async def get_momentum():
    """Get sentiment momentum indicator"""
    return {"momentum": get_momentum_sql(), "total_feedback": get_feedback_count_sql(), "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/city-risk")
async def get_city_risk():
    """Get city risk ranking leaderboard"""
    city_risks = get_city_risk_sql()
    return {
        "rankings": city_risks, "total_cities": len(city_risks),
        "critical_count": sum(1 for c in city_risks if c["status"] == "critical"),
        "watchlist_count": sum(1 for c in city_risks if c["status"] == "watchlist"),
        "stable_count": sum(1 for c in city_risks if c["status"] == "stable"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/anomalies")
async def get_anomalies():
    """Get detected anomalies"""
    anomalies = get_anomalies_sql()
    return {
        "anomalies": anomalies, "total_anomalies": len(anomalies),
        "high_severity": sum(1 for a in anomalies if a["severity"] == "high"),
        "medium_severity": sum(1 for a in anomalies if a["severity"] == "medium"),
        "low_severity": sum(1 for a in anomalies if a["severity"] == "low"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/export-report")
async def export_report():
    """Export intelligence report as JSON"""
    total = get_feedback_count_sql()
    
    if total == 0:
        return JSONResponse(content={"error": "No data available for export", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, status_code=200)
    
    summary = generate_ai_summary_sql()
    momentum = get_momentum_sql()
    city_risks = get_city_risk_sql()
    anomalies = get_anomalies_sql()
    actions = generate_governance_actions_sql()
    issue_dist = get_issue_distribution_sql()
    
    top_city = city_risks[0] if city_risks else {"city": "N/A", "risk_score": 0}
    top_issue = max(issue_dist, key=lambda k: issue_dist[k]) if issue_dist else "N/A"
    
    report = {
        "report_title": "BoothPulse AI - Governance Intelligence Report (SQL Edition)",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "database_engine": "SQLite",
        "summary": {"overall_risk_score": summary["risk_index"], "risk_level": "Low" if summary["risk_index"] < 30 else ("Moderate" if summary["risk_index"] < 60 else "High"), "total_feedback_analyzed": total, "sentiment_distribution": summary["stats"]},
        "momentum": {"direction": momentum["direction"], "change_percentage": momentum["change"], "assessment": "Improving" if momentum["direction"] == "improving" else ("Declining" if momentum["direction"] == "declining" else "Stable")},
        "top_risk_city": {"name": top_city["city"], "risk_score": top_city.get("risk_score", 0), "status": top_city.get("status", "unknown")},
        "top_issue": {"category": top_issue, "count": issue_dist.get(top_issue, 0)},
        "governance_actions": actions[:5],
        "anomalies_detected": len(anomalies),
        "primary_recommendation": summary["recommended_action"],
        "confidence_level": summary["confidence_level"],
        "city_risk_rankings": city_risks[:10],
        "full_summary": summary
    }
    
    return JSONResponse(content=report, headers={"Content-Disposition": f"attachment; filename=boothpulse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"})

@app.get("/raw-data")
async def get_raw_data(limit: int = Query(default=100, le=500)):
    """Get raw feedback data for table view"""
    return {"data": get_raw_data_sql(limit), "total": get_feedback_count_sql(), "limit": limit}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM feedback")
        feedback_count = cursor.fetchone()['cnt']
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM alerts")
        alert_count = cursor.fetchone()['cnt']
    
    return {
        "status": "healthy", "model_loaded": True, "version": "4.0.0",
        "database": "SQLite", "database_file": DATABASE_PATH,
        "total_feedback": feedback_count, "total_alerts": alert_count,
        "features": ["Structured SQL Database", "SQL Aggregation Analytics", "Governance Action Engine", "Sentiment Momentum Tracking", "City Risk Ranking", "Confidence-Weighted Risk Index", "Anomaly Detection", "Time Filter System", "Intelligence Export", "Raw Data Table View"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
