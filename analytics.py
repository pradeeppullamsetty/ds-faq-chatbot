#!/usr/bin/env python3
"""
Analytics module for the College FAQ Chatbot.

This module handles:
1. Logging user queries and responses to SQLite database
2. Managing analytics data with CRUD operations
3. Generating analytics data and statistics
4. Exporting analytics data to CSV

Technical implementation details:
- Uses SQLite for persistent storage
- Tracks query metadata, response times, user feedback
- Provides data retrieval for dashboard visualization
- Supports data export functionality
- Includes seed data generation for first-time users
"""
import os
import sqlite3
import json
import csv
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
import random

# Configuration
from config import COLLEGE_NAME

# Global database connection
_db_connection = None

# Analytics table schema
def get_analytics_table_schema() -> str:
    """Return SQL for creating the analytics table."""
    return """
CREATE TABLE IF NOT EXISTS analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    answered BOOLEAN NOT NULL,
    response_time REAL NOT NULL,
    feedback TEXT,
    language TEXT NOT NULL,
    sources_text TEXT,
    user_agent TEXT,
    session_id TEXT,
    source_document TEXT,
    page_number INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_analytics_timestamp (timestamp),
    INDEX idx_analytics_answered (answered),
    INDEX idx_analytics_date (created_at)
);
"""

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    global _db_connection

    if _db_connection is None:
        db_path = "analytics.db"
        _db_connection = sqlite3.connect(db_path)
        _db_connection.row_factory = sqlite3.Row
        _db_connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield _db_connection
    except Exception:
        _db_connection.rollback()
        raise
    finally:
        # Keep connection open for performance
        pass

def initialize_analytics_db():
    """Initialize analytics database and create tables."""
    try:
        with get_db_connection() as conn:
            # Create analytics table
            conn.execute(get_analytics_table_schema())

            # Create index for better performance
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_timestamp
            ON analytics (timestamp);
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_answered
            ON analytics (answered);
            """)

            conn.commit()

        print("Analytics database initialized successfully")
    except Exception as e:
        print(f"Error initializing analytics database: {str(e)}")
        raise

def log_query(
    question: str,
    answered: bool,
    response_time: float,
    feedback: Optional[str] = None,
    language: str = "en",
    sources_text: Optional[str] = None,
    user_agent: Optional[str] = None,
    session_id: Optional[str] = None,
    source_document: Optional[str] = None,
    page_number: Optional[int] = None
) -> int:
    """
    Log a user query and its response to the analytics database.

    Args:
        question: The user's question
        answered: Whether the question was answered
        response_time: Time taken to respond (in seconds)
        feedback: User feedback ("up" or "down")
        language: Language of the query ("en" or "ta")
        sources_text: Formatted source information
        user_agent: User's browser/device info
        session_id: Unique session identifier
        source_document: Source document filename
        page_number: Page number from document

    Returns:
        ID of the logged query
    """
    timestamp = datetime.now().isoformat()

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analytics (
                    timestamp, question, answered, response_time,
                    feedback, language, sources_text,
                    user_agent, session_id, source_document, page_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    question,
                    answered,
                    response_time,
                    feedback,
                    language,
                    sources_text,
                    user_agent,
                    session_id,
                    source_document,
                    page_number
                )
            )
            analytics_id = cursor.lastrowid
            conn.commit()

        return analytics_id

    except Exception as e:
        print(f"Error logging query to analytics: {str(e)}")
        raise

def get_analytics_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get analytics data with optional date filtering.

    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        limit: Maximum number of records to return

    Returns:
        List of analytics records as dictionaries
    """
    try:
        with get_db_connection() as conn:
            # Build query with optional date filtering
            query = "SELECT * FROM analytics WHERE 1=1"
            params = []

            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            # Convert rows to dictionaries
            analytics_data = []
            for row in rows:
                analytics_data.append(dict(row))

            return analytics_data

    except Exception as e:
        print(f"Error getting analytics data: {str(e)}")
        raise

def get_analytics_summary(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Get analytics summary statistics.

    Args:
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        Dictionary with analytics summary statistics
    """
    try:
        with get_db_connection() as conn:
            # Build query with optional date filtering
            base_query = "SELECT * FROM analytics WHERE 1=1"
            params = []

            if start_date:
                base_query += " AND created_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                base_query += " AND created_at <= ?"
                params.append(end_date.isoformat())

            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()

            if not rows:
                return get_empty_summary()

            # Calculate statistics
            total_queries = len(rows)
            answered_queries = sum(1 for row in rows if row["answered"])
            answer_rate = (answered_queries / total_queries * 100) if total_queries > 0 else 0

            # Response time statistics
            response_times = [row["response_time"] for row in rows]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            median_response_time = sorted(response_times)[len(response_times)//2] if response_times else 0

            # Feedback statistics
            feedback_up = sum(1 for row in rows if row["feedback"] == "up")
            feedback_down = sum(1 for row in rows if row["feedback"] == "down")
            satisfaction_rate = (feedback_up / (feedback_up + feedback_down) * 100) if (feedback_up + feedback_down) > 0 else 0

            # Questions per day
            dates = [datetime.fromisoformat(row["created_at"].split("T")[0]) for row in rows]
            questions_per_day = {}
            for date in dates:
                date_str = date.strftime("%Y-%m-%d")
                questions_per_day[date_str] = questions_per_day.get(date_str, 0) + 1

            # Top asked questions
            question_counts = {}
            for row in rows:
                question = row["question"]
                question_counts[question] = question_counts.get(question, 0) + 1

            top_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Unanswered questions for improvement
            unanswered_questions = [row["question"] for row in rows if not row["answered"]]

            return {
                "total_queries": total_queries,
                "answered_queries": answered_queries,
                "answer_rate": round(answer_rate, 2),
                "avg_response_time": round(avg_response_time, 3),
                "median_response_time": round(median_response_time, 3),
                "feedback_up": feedback_up,
                "feedback_down": feedback_down,
                "satisfaction_rate": round(satisfaction_rate, 2),
                "questions_per_day": questions_per_day,
                "top_questions": top_questions,
                "unanswered_questions": unanswered_questions[:5],  # Top 5 unanswered
                "date_range": {
                    "start": min(dates).isoformat() if dates else None,
                    "end": max(dates).isoformat() if dates else None
                }
            }

    except Exception as e:
        print(f"Error getting analytics summary: {str(e)}")
        raise

def get_empty_summary() -> Dict[str, Any]:
    """Return empty analytics summary when no data exists."""
    return {
        "total_queries": 0,
        "answered_queries": 0,
        "answer_rate": 0,
        "avg_response_time": 0,
        "median_response_time": 0,
        "feedback_up": 0,
        "feedback_down": 0,
        "satisfaction_rate": 0,
        "questions_per_day": {},
        "top_questions": [],
        "unanswered_questions": [],
        "date_range": {
            "start": None,
            "end": None
        }
    }

def export_analytics_to_csv(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, file_path: Optional[str] = None) -> str:
    """
    Export analytics data to CSV file.

    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        file_path: Path for CSV file (defaults to analytics_YYYY-MM-DD.csv)

    Returns:
        Path to the exported CSV file
    """
    if file_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = f"analytics_{date_str}.csv"

    try:
        analytics_data = get_analytics_data(start_date, end_date, limit=10000)

        if not analytics_data:
            raise ValueError("No analytics data to export")

        # Write to CSV
        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            if analytics_data:
                fieldnames = analytics_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(analytics_data)

        print(f"Analytics data exported to: {file_path}")
        return file_path

    except Exception as e:
        print(f"Error exporting analytics to CSV: {str(e)}")
        raise

def seed_demo_data():
    """Seed the analytics database with demo data for initial testing."""
    try:
        with get_db_connection() as conn:
            # Check if data already exists
            cursor = conn.execute("SELECT COUNT(*) as count FROM analytics")
            existing_count = cursor.fetchone()["count"]

            if existing_count > 0:
                print(f"Demo data already exists ({existing_count} records). Skipping seeding.")
                return

            # Generate demo data
            print("Seeding demo data for analytics...")

            # Questions and answers for demo
            demo_questions = [
                ("What are the admission requirements for B.Tech?", True),
                ("How to apply for scholarships?", True),
                ("What are the hostel facilities available?", True),
                ("What courses are offered in Computer Science?", True),
                ("What is the placement record of this college?", True),
                ("What are the fees for MBA program?", True),
                ("When is the application deadline for 2024?", False),
                ("How to get a student visa?", False),
                ("What research papers are available in the library?", False),
                ("When is the next college festival?", False),
            ]

            # Generate timestamps for the last 30 days
            base_time = datetime.now()

            for i, (question, answered) in enumerate(demo_questions):
                # Calculate timestamp
                hours_ago = i * 24 * 3600 // len(demo_questions)  # Spread over 24 hours
                timestamp = base_time - timedelta(hours=hours_ago)

                # Generate random response time
                response_time = round(random.uniform(0.5, 3.0), 3)

                # Generate feedback
                if answered:
                    feedback = random.choice(["up", "up", "up", "down", None])
                else:
                    feedback = None

                # Language
                language = random.choice(["en", "ta"])

                # Insert record
                conn.execute(
                    """
                    INSERT INTO analytics (
                        timestamp, question, answered, response_time,
                        feedback, language, sources_text,
                        user_agent, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp.isoformat(),
                        question,
                        answered,
                        response_time,
                        feedback,
                        language,
                        f"Source {i+1}: demo_content.txt, Page {i+1}: Sample answer..." if answered else None,
                        "Streamlit/1.0.0",
                        f"demo_session_{i:04d}"
                    )
                )

            conn.commit()
            print(f"Demo data seeded: {len(demo_questions)} records added")

    except Exception as e:
        print(f"Error seeding demo data: {str(e)}")
        raise

def get_analytics_stats() -> Dict[str, Any]:
    """
    Get comprehensive analytics statistics.

    Args:
        None

    Returns:
        Dictionary with analytics statistics
    """
    try:
        with get_db_connection() as conn:
            # Total records
            cursor = conn.execute("SELECT COUNT(*) as count FROM analytics")
            total_records = cursor.fetchone()["count"]

            # Records from today
            today = datetime.now().strftime("%Y-%m-%d")
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM analytics WHERE DATE(created_at) = ?",
                (today,)
            )
            today_records = cursor.fetchone()["count"]

            # Answered vs Unanswered
            cursor = conn.execute("SELECT answered, COUNT(*) as count FROM analytics GROUP BY answered")
            answered_stats = {row["answered"]: row["count"] for row in cursor.fetchall()}

            # Feedback statistics
            cursor = conn.execute(
                "SELECT feedback, COUNT(*) as count FROM analytics WHERE feedback IS NOT NULL GROUP BY feedback"
            )
            feedback_stats = {row["feedback"] or "neutral": row["count"] for row in cursor.fetchall()}

            # Language statistics
            cursor = conn.execute("SELECT language, COUNT(*) as count FROM analytics GROUP BY language")
            language_stats = {row["language"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_records": total_records,
                "today_records": today_records,
                "answered_stats": answered_stats,
                "feedback_stats": feedback_stats,
                "language_stats": language_stats
            }

    except Exception as e:
        print(f"Error getting analytics stats: {str(e)}")
        raise

def clear_analytics_data():
    """Clear all analytics data and reseed demo data."""
    try:
        with get_db_connection() as conn:
            # Clear all data
            conn.execute("DELETE FROM analytics")
            conn.commit()

            print("Analytics data cleared")

            # Reseed demo data
            seed_demo_data()

            print("Demo data reseeded successfully")

    except Exception as e:
        print(f"Error clearing analytics data: {str(e)}")
        raise

def cleanup():
    """Clean up database connections."""
    global _db_connection

    if _db_connection:
        _db_connection.close()
        _db_connection = None
        print("Analytics database connection closed")