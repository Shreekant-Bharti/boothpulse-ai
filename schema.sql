-- BoothPulse AI - Geo-Intelligent Governance Intelligence Engine
-- Database Schema v3.0.0
-- SQLite Database Schema

-- Main feedback table for storing citizen feedback
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    sentiment TEXT CHECK(sentiment IN ('positive', 'negative', 'neutral')),
    sentiment_score REAL,
    issue TEXT CHECK(issue IN ('Water', 'Roads', 'Employment', 'Healthcare', 'Education', 'General')),
    country TEXT DEFAULT 'India',
    state TEXT,
    city TEXT,
    latitude REAL,
    longitude REAL,
    confidence REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table for storing system alerts
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    detail TEXT,
    severity TEXT CHECK(severity IN ('high', 'medium', 'low')) DEFAULT 'medium',
    alert_type TEXT,
    acknowledged INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_feedback_sentiment ON feedback(sentiment);
CREATE INDEX IF NOT EXISTS idx_feedback_issue ON feedback(issue);
CREATE INDEX IF NOT EXISTS idx_feedback_city ON feedback(city);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);

-- View for city risk aggregation (used by dashboard)
CREATE VIEW IF NOT EXISTS v_city_risk AS
SELECT 
    city,
    COUNT(*) as total,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
    AVG(confidence) as avg_confidence,
    ROUND(
        CAST(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS FLOAT) / 
        NULLIF(COUNT(*), 0) * 100, 1
    ) as negative_ratio,
    ROUND(
        (CAST(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) AS FLOAT) / 
        NULLIF(COUNT(*), 0)) * AVG(confidence) * MIN(1.0, COUNT(*) / 20.0) * 100, 0
    ) as risk_score
FROM feedback
WHERE city IS NOT NULL AND city != ''
GROUP BY city;

-- View for issue distribution
CREATE VIEW IF NOT EXISTS v_issue_distribution AS
SELECT 
    issue,
    COUNT(*) as count,
    ROUND(CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM feedback) * 100, 1) as percentage
FROM feedback
GROUP BY issue;

-- View for sentiment distribution
CREATE VIEW IF NOT EXISTS v_sentiment_distribution AS
SELECT 
    sentiment,
    COUNT(*) as count,
    ROUND(CAST(COUNT(*) AS FLOAT) / NULLIF((SELECT COUNT(*) FROM feedback), 0) * 100, 1) as percentage
FROM feedback
GROUP BY sentiment;
