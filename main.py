"""
BoothPulse AI - Geo-Intelligent Governance Intelligence Engine
Advanced booth-level sentiment analysis with geo-intelligence capabilities
Hackathon Finalist Ready - Production Grade
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import httpx
import asyncio
import os

# Initialize FastAPI app
app = FastAPI(
    title="BoothPulse AI",
    description="Geo-Intelligent Governance Intelligence Engine",
    version="2.0.0"
)

# Mount static files
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load VADER sentiment analyzer (lightweight, ~1MB)
print("🚀 Loading VADER sentiment analysis model...")
sentiment_analyzer = SentimentIntensityAnalyzer()
print("✅ Model loaded successfully!")

# ==================== GEO DATA ====================

COUNTRIES = {
    "India": "IN",
    "USA": "US",
    "UK": "GB",
    "Canada": "CA",
    "Australia": "AU",
    "Other": "OT"
}

STATES_BY_COUNTRY = {
    "India": [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
        "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Delhi", "Chandigarh", "Puducherry"
    ],
    "USA": [
        "California", "Texas", "Florida", "New York", "Pennsylvania", "Illinois",
        "Ohio", "Georgia", "North Carolina", "Michigan", "Washington", "Arizona",
        "Massachusetts", "Tennessee", "Indiana", "Missouri", "Maryland", "Wisconsin"
    ],
    "UK": [
        "England", "Scotland", "Wales", "Northern Ireland", "Greater London",
        "West Midlands", "Greater Manchester", "West Yorkshire", "South Yorkshire"
    ],
    "Canada": [
        "Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba",
        "Saskatchewan", "Nova Scotia", "New Brunswick", "Newfoundland"
    ],
    "Australia": [
        "New South Wales", "Victoria", "Queensland", "Western Australia",
        "South Australia", "Tasmania", "Northern Territory", "Australian Capital Territory"
    ],
    "Other": ["State 1", "State 2", "State 3", "State 4", "State 5"]
}

CITIES_BY_STATE = {
    # India
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
    # USA
    "California": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento"],
    "Texas": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"],
    "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville", "Naples"],
    "New York": ["New York City", "Buffalo", "Rochester", "Albany", "Syracuse"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Erie"],
    "Illinois": ["Chicago", "Aurora", "Naperville", "Rockford"],
    "Ohio": ["Columbus", "Cleveland", "Cincinnati", "Toledo"],
    "Georgia": ["Atlanta", "Augusta", "Savannah", "Columbus"],
    "Washington": ["Seattle", "Spokane", "Tacoma", "Vancouver"],
    "Arizona": ["Phoenix", "Tucson", "Mesa", "Scottsdale"],
    "Massachusetts": ["Boston", "Worcester", "Springfield", "Cambridge"],
    # UK
    "England": ["London", "Manchester", "Birmingham", "Liverpool", "Leeds", "Bristol"],
    "Scotland": ["Edinburgh", "Glasgow", "Aberdeen", "Dundee"],
    "Wales": ["Cardiff", "Swansea", "Newport", "Wrexham"],
    "Greater London": ["Westminster", "Camden", "Greenwich", "Hackney", "Kensington"],
    "Northern Ireland": ["Belfast", "Derry", "Lisburn", "Newry"],
    # Canada
    "Ontario": ["Toronto", "Ottawa", "Mississauga", "Hamilton", "London", "Brampton"],
    "Quebec": ["Montreal", "Quebec City", "Laval", "Gatineau", "Longueuil"],
    "British Columbia": ["Vancouver", "Victoria", "Surrey", "Burnaby", "Richmond"],
    "Alberta": ["Calgary", "Edmonton", "Red Deer", "Lethbridge"],
    "Manitoba": ["Winnipeg", "Brandon", "Steinbach"],
    # Australia
    "New South Wales": ["Sydney", "Newcastle", "Wollongong", "Central Coast"],
    "Victoria": ["Melbourne", "Geelong", "Ballarat", "Bendigo"],
    "Queensland": ["Brisbane", "Gold Coast", "Cairns", "Townsville", "Sunshine Coast"],
    "Western Australia": ["Perth", "Fremantle", "Bunbury", "Geraldton"],
    "South Australia": ["Adelaide", "Mount Gambier", "Whyalla"],
    # Other
    "State 1": ["City A", "City B", "City C"],
    "State 2": ["City D", "City E", "City F"],
    "State 3": ["City G", "City H", "City I"],
    "State 4": ["City J", "City K", "City L"],
    "State 5": ["City M", "City N", "City O"],
}

# Predefined city coordinates
CITY_COORDINATES = {
    # India
    "Mumbai": {"lat": 19.0760, "lng": 72.8777},
    "Delhi": {"lat": 28.6139, "lng": 77.2090},
    "New Delhi": {"lat": 28.6139, "lng": 77.2090},
    "Bengaluru": {"lat": 12.9716, "lng": 77.5946},
    "Chennai": {"lat": 13.0827, "lng": 80.2707},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lng": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lng": 72.5714},
    "Pune": {"lat": 18.5204, "lng": 73.8567},
    "Jaipur": {"lat": 26.9124, "lng": 75.7873},
    "Lucknow": {"lat": 26.8467, "lng": 80.9462},
    "Surat": {"lat": 21.1702, "lng": 72.8311},
    "Bhopal": {"lat": 23.2599, "lng": 77.4126},
    "Patna": {"lat": 25.5941, "lng": 85.1376},
    "Chandigarh": {"lat": 30.7333, "lng": 76.7794},
    "Coimbatore": {"lat": 11.0168, "lng": 76.9558},
    "Indore": {"lat": 22.7196, "lng": 75.8577},
    "Nagpur": {"lat": 21.1458, "lng": 79.0882},
    "Kochi": {"lat": 9.9312, "lng": 76.2673},
    "Shimla": {"lat": 31.1048, "lng": 77.1734},
    "Guwahati": {"lat": 26.1445, "lng": 91.7362},
    "Thiruvananthapuram": {"lat": 8.5241, "lng": 76.9366},
    "Ranchi": {"lat": 23.3441, "lng": 85.3096},
    "Bhubaneswar": {"lat": 20.2961, "lng": 85.8245},
    "Panaji": {"lat": 15.4909, "lng": 73.8278},
    "Gurugram": {"lat": 28.4595, "lng": 77.0266},
    "Noida": {"lat": 28.5355, "lng": 77.3910},
    "Visakhapatnam": {"lat": 17.6868, "lng": 83.2185},
    "Vijayawada": {"lat": 16.5062, "lng": 80.6480},
    "Raipur": {"lat": 21.2514, "lng": 81.6296},
    "Dehradun": {"lat": 30.3165, "lng": 78.0322},
    "Varanasi": {"lat": 25.3176, "lng": 82.9739},
    "Agra": {"lat": 27.1767, "lng": 78.0081},
    "Mysuru": {"lat": 12.2958, "lng": 76.6394},
    "Madurai": {"lat": 9.9252, "lng": 78.1198},
    # USA
    "New York City": {"lat": 40.7128, "lng": -74.0060},
    "Los Angeles": {"lat": 34.0522, "lng": -118.2437},
    "Chicago": {"lat": 41.8781, "lng": -87.6298},
    "Houston": {"lat": 29.7604, "lng": -95.3698},
    "San Francisco": {"lat": 37.7749, "lng": -122.4194},
    "Miami": {"lat": 25.7617, "lng": -80.1918},
    "Dallas": {"lat": 32.7767, "lng": -96.7970},
    "Austin": {"lat": 30.2672, "lng": -97.7431},
    "Seattle": {"lat": 47.6062, "lng": -122.3321},
    "Boston": {"lat": 42.3601, "lng": -71.0589},
    "Atlanta": {"lat": 33.7490, "lng": -84.3880},
    "Phoenix": {"lat": 33.4484, "lng": -112.0740},
    # UK
    "London": {"lat": 51.5074, "lng": -0.1278},
    "Manchester": {"lat": 53.4808, "lng": -2.2426},
    "Birmingham": {"lat": 52.4862, "lng": -1.8904},
    "Edinburgh": {"lat": 55.9533, "lng": -3.1883},
    "Glasgow": {"lat": 55.8642, "lng": -4.2518},
    "Liverpool": {"lat": 53.4084, "lng": -2.9916},
    "Belfast": {"lat": 54.5973, "lng": -5.9301},
    "Cardiff": {"lat": 51.4816, "lng": -3.1791},
    # Canada
    "Toronto": {"lat": 43.6532, "lng": -79.3832},
    "Vancouver": {"lat": 49.2827, "lng": -123.1207},
    "Montreal": {"lat": 45.5017, "lng": -73.5673},
    "Calgary": {"lat": 51.0447, "lng": -114.0719},
    "Edmonton": {"lat": 53.5461, "lng": -113.4938},
    "Ottawa": {"lat": 45.4215, "lng": -75.6972},
    "Winnipeg": {"lat": 49.8951, "lng": -97.1384},
    # Australia
    "Sydney": {"lat": -33.8688, "lng": 151.2093},
    "Melbourne": {"lat": -37.8136, "lng": 144.9631},
    "Brisbane": {"lat": -27.4698, "lng": 153.0251},
    "Perth": {"lat": -31.9505, "lng": 115.8605},
    "Adelaide": {"lat": -34.9285, "lng": 138.6007},
    "Gold Coast": {"lat": -28.0167, "lng": 153.4000},
}

# Issue keywords mapping
ISSUE_KEYWORDS = {
    "Water": ["water", "drinking", "supply", "tank", "pipe", "drainage", "tap", "well", "borewell", "shortage", "sewage", "flood", "sanitation"],
    "Roads": ["road", "pothole", "street", "highway", "traffic", "footpath", "bridge", "construction", "pavement", "transport", "accident", "signal"],
    "Employment": ["job", "employment", "work", "salary", "unemployment", "hiring", "career", "wages", "labor", "income", "livelihood", "business"],
    "Healthcare": ["hospital", "health", "doctor", "medicine", "clinic", "medical", "treatment", "disease", "nurse", "vaccine", "ambulance", "pharmacy"],
    "Education": ["school", "education", "college", "teacher", "student", "learning", "exam", "university", "books", "scholarship", "literacy", "tuition"]
}

# ==================== IN-MEMORY STORAGE ====================

feedback_data: List[Dict[str, Any]] = []
alerts: List[Dict[str, Any]] = []
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

class GeoLocation(BaseModel):
    country: str
    state: str
    city: str

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
        return max(scores, key=scores.get)
    return "General"


def map_sentiment(label: str, score: float) -> tuple:
    """Map HuggingFace sentiment to categories with refined thresholds"""
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


def calculate_sentiment_risk_index(feedback_list: List[Dict]) -> int:
    """Calculate Sentiment Risk Index (0-100)"""
    if not feedback_list:
        return 0
    
    negative_count = sum(1 for f in feedback_list if f["sentiment"] == "negative")
    total = len(feedback_list)
    
    risk_index = int((negative_count / total) * 100)
    return min(risk_index, 100)


def check_escalation_risk(city: str, feedback_list: List[Dict]) -> Optional[Dict]:
    """Check for escalation risk in a city"""
    city_feedback = [f for f in feedback_list if f["city"] == city]
    
    if len(city_feedback) < 3:
        return None
    
    recent_feedback = city_feedback[-10:]
    negative_count = sum(1 for f in recent_feedback if f["sentiment"] == "negative")
    negative_ratio = negative_count / len(recent_feedback)
    
    # Check conditions
    if negative_ratio > 0.4 or negative_count >= 5:
        severity = "high" if negative_ratio > 0.6 else ("medium" if negative_ratio > 0.4 else "low")
        return {
            "city": city,
            "negative_ratio": round(negative_ratio * 100, 1),
            "negative_count": negative_count,
            "severity": severity
        }
    
    return None


def generate_ai_summary(feedback_list: List[Dict]) -> str:
    """Generate AI Intelligence Summary"""
    if not feedback_list:
        return "No data available yet. Submit feedback to generate intelligence insights."
    
    total = len(feedback_list)
    
    # Calculate sentiment distribution
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for f in feedback_list:
        sentiment_counts[f["sentiment"]] += 1
    
    positive_pct = round((sentiment_counts["positive"] / total) * 100, 1)
    negative_pct = round((sentiment_counts["negative"] / total) * 100, 1)
    neutral_pct = round((sentiment_counts["neutral"] / total) * 100, 1)
    
    # Find dominant sentiment
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    
    # Find most common issue
    issue_counts = defaultdict(int)
    for f in feedback_list:
        issue_counts[f["issue"]] += 1
    
    top_issue = max(issue_counts, key=issue_counts.get) if issue_counts else "General"
    top_issue_count = issue_counts[top_issue]
    
    # Find most active city
    city_counts = defaultdict(int)
    city_sentiment = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for f in feedback_list:
        city_counts[f["city"]] += 1
        city_sentiment[f["city"]][f["sentiment"]] += 1
    
    most_active_city = max(city_counts, key=city_counts.get) if city_counts else "Unknown"
    city_negative_pct = 0
    if most_active_city in city_sentiment:
        city_total = sum(city_sentiment[most_active_city].values())
        if city_total > 0:
            city_negative_pct = round((city_sentiment[most_active_city]["negative"] / city_total) * 100, 1)
    
    # Calculate risk level
    risk_index = calculate_sentiment_risk_index(feedback_list)
    risk_level = "low" if risk_index < 30 else ("moderate" if risk_index < 60 else "high")
    
    # Generate summary
    summary_parts = []
    
    summary_parts.append(f"📊 **Overall Analysis**: Processed {total} feedback entries across multiple regions.")
    
    if dominant_sentiment == "negative":
        summary_parts.append(f"⚠️ **Sentiment Alert**: Overall sentiment is {dominant_sentiment} ({negative_pct}%). Immediate attention recommended.")
    elif dominant_sentiment == "positive":
        summary_parts.append(f"✅ **Sentiment Status**: Overall sentiment is {dominant_sentiment} ({positive_pct}%). Public satisfaction is healthy.")
    else:
        summary_parts.append(f"📈 **Sentiment Status**: Overall sentiment is mixed ({neutral_pct}% neutral). Monitoring recommended.")
    
    summary_parts.append(f"🎯 **Key Issue**: {top_issue} ({top_issue_count} reports) is the primary concern requiring attention.")
    
    summary_parts.append(f"📍 **Focus Area**: {most_active_city} shows {city_negative_pct}% negative feedback ratio.")
    
    if risk_level == "high":
        summary_parts.append(f"🚨 **Risk Assessment**: Escalation risk is {risk_level} (Index: {risk_index}/100). Immediate intervention advised.")
    elif risk_level == "moderate":
        summary_parts.append(f"⚡ **Risk Assessment**: Escalation risk is {risk_level} (Index: {risk_index}/100). Close monitoring needed.")
    else:
        summary_parts.append(f"🟢 **Risk Assessment**: Escalation risk is {risk_level} (Index: {risk_index}/100). Situation stable.")
    
    return " ".join(summary_parts)


async def resolve_geo_location(street: str, landmark: str, city: str, state: str, country: str) -> Optional[Dict[str, float]]:
    """Resolve coordinates using OpenStreetMap Nominatim API"""
    # Build query
    query_parts = []
    if street:
        query_parts.append(street)
    if landmark:
        query_parts.append(landmark)
    query_parts.append(city)
    query_parts.append(state)
    query_parts.append(country)
    
    query = ", ".join(filter(None, query_parts))
    
    # Check cache
    if query in geo_cache:
        return geo_cache[query]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": query,
                    "format": "json",
                    "limit": 1
                },
                headers={"User-Agent": "BoothPulseAI/2.0"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = {
                        "lat": float(data[0]["lat"]),
                        "lon": float(data[0]["lon"])
                    }
                    geo_cache[query] = result
                    return result
    except Exception as e:
        print(f"Geo resolution error: {e}")
    
    return None


def create_alert(city: str, state: str, country: str, negative_ratio: float, severity: str) -> Dict:
    """Create an alert entry"""
    alert = {
        "id": len(alerts) + 1,
        "city": city,
        "state": state,
        "country": country,
        "message": f"🚨 Escalation Risk Detected in {city}, {state}",
        "detail": f"Negative sentiment ratio: {negative_ratio}%",
        "severity": severity,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "acknowledged": False
    }
    alerts.append(alert)
    return alert

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/geo-data")
async def get_geo_data():
    """Get countries, states, and cities data"""
    return {
        "countries": list(COUNTRIES.keys()),
        "states_by_country": STATES_BY_COUNTRY,
        "cities_by_state": CITIES_BY_STATE
    }


@app.post("/analyze")
async def analyze_feedback(feedback: FeedbackInput):
    """Analyze sentiment of submitted feedback with geo-intelligence"""
    text = feedback.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Feedback text is required")
    
    # Analyze sentiment using VADER
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # Map VADER compound score to sentiment label
    if compound >= 0.05:
        raw_label = "POSITIVE"
        raw_score = (compound + 1) / 2  # Normalize to 0-1
    elif compound <= -0.05:
        raw_label = "NEGATIVE"
        raw_score = (-compound + 1) / 2  # Normalize to 0-1
    else:
        raw_label = "NEUTRAL"
        raw_score = 0.5
    
    # Map sentiment
    sentiment, confidence = map_sentiment(raw_label, raw_score)
    
    # Detect issue category
    issue = detect_issue_category(text)
    
    # Handle geo location
    lat = feedback.lat
    lon = feedback.lon
    
    # Try to resolve geo if manual location provided
    if (feedback.street or feedback.landmark) and (not lat or not lon):
        resolved = await resolve_geo_location(
            feedback.street or "",
            feedback.landmark or "",
            feedback.city,
            feedback.state,
            feedback.country
        )
        if resolved:
            lat = resolved["lat"]
            lon = resolved["lon"]
    
    # Fallback to city coordinates
    if (not lat or not lon) and feedback.city in CITY_COORDINATES:
        coords = CITY_COORDINATES[feedback.city]
        lat = coords["lat"]
        lon = coords["lng"]
    
    # Default coordinates (center of India)
    if not lat or not lon:
        lat = 20.5937
        lon = 78.9629
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create feedback entry
    entry = {
        "id": len(feedback_data) + 1,
        "text": text,
        "sentiment": sentiment,
        "issue": issue,
        "country": feedback.country,
        "state": feedback.state,
        "city": feedback.city,
        "lat": lat,
        "lon": lon,
        "confidence": round(confidence, 4),
        "timestamp": timestamp,
        "ward": feedback.ward,
        "booth_number": feedback.booth_number
    }
    
    feedback_data.append(entry)
    
    # Check for escalation risk
    alert_triggered = False
    escalation = check_escalation_risk(feedback.city, feedback_data)
    if escalation and escalation["severity"] in ["medium", "high"]:
        # Check if alert already exists for this city recently
        existing_alert = any(
            a["city"] == feedback.city and 
            (datetime.now() - datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")).seconds < 300
            for a in alerts[-10:]
        )
        if not existing_alert:
            create_alert(
                feedback.city,
                feedback.state,
                feedback.country,
                escalation["negative_ratio"],
                escalation["severity"]
            )
            alert_triggered = True
    
    return {
        "success": True,
        "data": entry,
        "alert_triggered": alert_triggered,
        "message": f"Feedback analyzed: {sentiment.upper()} sentiment detected"
    }


@app.get("/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    total = len(feedback_data)
    
    if total == 0:
        return {
            "total_feedback": 0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "positive_pct": 0,
            "negative_pct": 0,
            "neutral_pct": 0,
            "risk_index": 0,
            "early_warning": False,
            "geo_data": [],
            "issue_distribution": {},
            "timeline_data": [],
            "recent_activity": [],
            "city_stats": []
        }
    
    # Sentiment distribution
    sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
    for entry in feedback_data:
        sentiment_dist[entry["sentiment"]] += 1
    
    positive_pct = round((sentiment_dist["positive"] / total) * 100, 1)
    negative_pct = round((sentiment_dist["negative"] / total) * 100, 1)
    neutral_pct = round((sentiment_dist["neutral"] / total) * 100, 1)
    
    # Risk index
    risk_index = calculate_sentiment_risk_index(feedback_data)
    
    # Early warning
    early_warning = negative_pct > 40 or risk_index > 60
    
    # Geo data for map
    geo_data = [
        {
            "lat": f["lat"],
            "lon": f["lon"],
            "sentiment": f["sentiment"],
            "confidence": f["confidence"],
            "city": f["city"],
            "issue": f["issue"]
        }
        for f in feedback_data if f.get("lat") and f.get("lon")
    ]
    
    # Issue distribution
    issue_counts = defaultdict(int)
    for f in feedback_data:
        issue_counts[f["issue"]] += 1
    
    issue_distribution = {
        "Water": issue_counts.get("Water", 0),
        "Roads": issue_counts.get("Roads", 0),
        "Employment": issue_counts.get("Employment", 0),
        "Healthcare": issue_counts.get("Healthcare", 0),
        "Education": issue_counts.get("Education", 0),
        "General": issue_counts.get("General", 0)
    }
    
    # Timeline data (last 20 entries for trend)
    timeline_data = []
    for f in feedback_data[-20:]:
        score = 1 if f["sentiment"] == "positive" else (-1 if f["sentiment"] == "negative" else 0)
        timeline_data.append({
            "timestamp": f["timestamp"],
            "score": score,
            "sentiment": f["sentiment"]
        })
    
    # City stats
    city_sentiment = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
    for f in feedback_data:
        city = f["city"]
        city_sentiment[city][f["sentiment"]] += 1
        city_sentiment[city]["total"] += 1
    
    city_stats = [
        {
            "city": city,
            "positive": stats["positive"],
            "negative": stats["negative"],
            "neutral": stats["neutral"],
            "total": stats["total"],
            "negative_ratio": round((stats["negative"] / stats["total"]) * 100, 1) if stats["total"] > 0 else 0
        }
        for city, stats in city_sentiment.items()
    ]
    city_stats.sort(key=lambda x: x["negative_ratio"], reverse=True)
    
    # Recent activity
    recent_activity = feedback_data[-10:][::-1]
    
    return {
        "total_feedback": total,
        "sentiment_distribution": sentiment_dist,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "neutral_pct": neutral_pct,
        "risk_index": risk_index,
        "early_warning": early_warning,
        "geo_data": geo_data,
        "issue_distribution": issue_distribution,
        "timeline_data": timeline_data,
        "recent_activity": recent_activity,
        "city_stats": city_stats[:10]
    }


@app.get("/alerts")
async def get_alerts():
    """Get all alerts with filtering options"""
    return {
        "alerts": alerts[-50:][::-1],
        "total": len(alerts),
        "unacknowledged": sum(1 for a in alerts if not a["acknowledged"]),
        "high_severity": sum(1 for a in alerts if a["severity"] == "high"),
        "medium_severity": sum(1 for a in alerts if a["severity"] == "medium"),
        "low_severity": sum(1 for a in alerts if a["severity"] == "low")
    }


@app.get("/ai-summary")
async def get_ai_summary():
    """Get AI-generated intelligence summary"""
    summary = generate_ai_summary(feedback_data)
    
    # Additional metrics
    risk_index = calculate_sentiment_risk_index(feedback_data)
    risk_level = "low" if risk_index < 30 else ("moderate" if risk_index < 60 else "high")
    
    # Cities at risk
    cities_at_risk = []
    city_groups = defaultdict(list)
    for f in feedback_data:
        city_groups[f["city"]].append(f)
    
    for city, city_feedback in city_groups.items():
        escalation = check_escalation_risk(city, feedback_data)
        if escalation:
            cities_at_risk.append(escalation)
    
    return {
        "summary": summary,
        "risk_index": risk_index,
        "risk_level": risk_level,
        "total_feedback": len(feedback_data),
        "cities_at_risk": cities_at_risk,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "2.0.0",
        "total_feedback": len(feedback_data),
        "total_alerts": len(alerts)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
