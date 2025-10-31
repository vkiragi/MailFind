"""
Natural language query parser for email search.
Converts queries like "all updates from john in the last 3 days" into structured JSON.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from openai import OpenAI
from dateutil import parser as date_parser
import os


class QueryParser:
    """Parse natural language email queries into structured format."""

    # Few-shot examples for the LLM
    FEW_SHOT_EXAMPLES = """
Example 1:
Query: "all updates from john in the last 3 days"
Output: {"intent": "fetch_emails", "people": ["john"], "time_range": {"last_n": "3 days"}, "topics": [], "limits": {"k": 50}}

Example 2:
Query: "show me receipts from DoorDash this month"
Output: {"intent": "list_receipts", "people": ["DoorDash"], "time_range": {"last_n": "1 month"}, "topics": ["receipt", "order"], "limits": {"k": 100}}

Example 3:
Query: "summarize all emails about the contract negotiation"
Output: {"intent": "summarize", "people": [], "time_range": {}, "topics": ["contract", "negotiation"], "limits": {"k": 50}}

Example 4:
Query: "flight confirmations from United Airlines"
Output: {"intent": "fetch_emails", "people": ["United Airlines"], "time_range": {}, "topics": ["flight", "confirmation"], "limits": {"k": 50}}

Example 5:
Query: "emails from sarah about the project in the last week"
Output: {"intent": "fetch_emails", "people": ["sarah"], "time_range": {"last_n": "1 week"}, "topics": ["project"], "limits": {"k": 50}}

Example 6:
Query: "all invoices over $500 since January"
Output: {"intent": "fetch_emails", "people": [], "time_range": {"since": "2025-01-01T00:00:00Z"}, "topics": ["invoice"], "filters": {"amount_min": 500}, "limits": {"k": 100}}

Example 7:
Query: "meeting notes from yesterday"
Output: {"intent": "fetch_emails", "people": [], "time_range": {"last_n": "1 day"}, "topics": ["meeting", "notes"], "limits": {"k": 20}}

Example 8:
Query: "show me the last 10 emails from my boss"
Output: {"intent": "fetch_emails", "people": ["boss"], "time_range": {}, "topics": [], "limits": {"k": 10}}

Example 9:
Query: "package delivery notifications this week"
Output: {"intent": "fetch_emails", "people": [], "time_range": {"last_n": "1 week"}, "topics": ["package", "delivery", "notification"], "limits": {"k": 50}}

Example 10:
Query: "all Amazon orders between March and May"
Output: {"intent": "fetch_emails", "people": ["Amazon"], "time_range": {"since": "2025-03-01T00:00:00Z", "until": "2025-05-31T23:59:59Z"}, "topics": ["order"], "limits": {"k": 100}}

Example 11:
Query: "what did alice say about the bug fix?"
Output: {"intent": "summarize", "people": ["alice"], "time_range": {}, "topics": ["bug", "fix"], "limits": {"k": 20}}

Example 12:
Query: "latest 5 newsletters from TechCrunch"
Output: {"intent": "fetch_emails", "people": ["TechCrunch"], "time_range": {}, "topics": ["newsletter"], "limits": {"k": 5}}

Example 13:
Query: "password reset emails from the last 2 hours"
Output: {"intent": "fetch_emails", "people": [], "time_range": {"last_n": "2 hours"}, "topics": ["password", "reset"], "limits": {"k": 10}}

Example 14:
Query: "all expense reports I sent last quarter"
Output: {"intent": "fetch_emails", "people": [], "time_range": {"last_n": "1 quarter"}, "topics": ["expense", "report"], "filters": {"sent_by_me": true}, "limits": {"k": 50}}

Example 15:
Query: "show unread messages from the support team"
Output: {"intent": "fetch_emails", "people": ["support team"], "time_range": {}, "topics": [], "filters": {"unread": true}, "limits": {"k": 50}}
"""

    SYSTEM_PROMPT = """You extract a normalized query for email retrieval based on natural language input.

Rules:
1. Use "fetch_emails" for retrieval, "summarize" for summary requests, "list_receipts" for transaction/receipt queries
2. Extract people names exactly as mentioned (we'll normalize later)
3. For relative dates use last_n, for absolute dates use since/until in ISO8601
4. Extract topic keywords from the query
5. Set sensible default limits (50 for most, 100 for receipts, 10-20 for recent queries)
6. Only include filters that are explicitly mentioned

Examples:
- "all updates from john in the last 3 days" → intent: fetch_emails, people: ["john"], time_range: {last_n: "3 days"}
- "show me receipts from DoorDash this month" → intent: list_receipts, people: ["DoorDash"], topics: ["receipt", "order"], time_range: {last_n: "1 month"}
- "summarize emails about the contract" → intent: summarize, topics: ["contract"]
- "emails from sarah about the project in the last week" → intent: fetch_emails, people: ["sarah"], topics: ["project"], time_range: {last_n: "1 week"}
"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize parser with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key)

    def parse(self, query: str, db_connection=None) -> Dict[str, Any]:
        """
        Parse a natural language query into structured format.

        Args:
            query: Natural language query string
            db_connection: Optional database connection for contact normalization

        Returns:
            Dictionary with structured query parameters
        """
        # First try LLM-based extraction
        try:
            structured_query = self._parse_with_llm(query)
        except Exception as e:
            print(f"LLM parsing failed: {e}, falling back to regex")
            structured_query = self._parse_with_regex(query)

        # Post-process with deterministic helpers
        structured_query = self._normalize_dates(structured_query)

        if db_connection:
            structured_query = self._normalize_people(structured_query, db_connection)

        return structured_query

    def _parse_with_llm(self, query: str) -> Dict[str, Any]:
        """Use OpenAI API with structured output (JSON schema mode)."""

        # Define the JSON schema for structured output
        response_schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["fetch_emails", "summarize", "list_receipts"],
                    "description": "The user's intent"
                },
                "people": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names, emails, or organizations mentioned"
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords or subject themes"
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "since": {"type": "string", "description": "ISO8601 start date"},
                        "until": {"type": "string", "description": "ISO8601 end date"},
                        "last_n": {"type": "string", "description": "Relative time like '3 days'"}
                    },
                    "required": [],
                    "additionalProperties": False
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "amount_min": {"type": "number"},
                        "amount_max": {"type": "number"},
                        "unread": {"type": "boolean"},
                        "sent_by_me": {"type": "boolean"},
                        "has_attachment": {"type": "boolean"}
                    },
                    "required": [],
                    "additionalProperties": False
                },
                "limits": {
                    "type": "object",
                    "properties": {
                        "k": {"type": "integer", "description": "Max results"}
                    },
                    "required": ["k"],
                    "additionalProperties": False
                }
            },
            "required": ["intent", "people", "topics", "time_range", "limits"],
            "additionalProperties": False
        }

        # Use GPT-4o-mini for cost efficiency (or gpt-4o for better accuracy)
        # Note: GPT-5 nano isn't available yet, but this will work when it is
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Change to "gpt-4o" or "gpt-5-nano" when available
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT + "\n\nFew-shot examples:\n" + self.FEW_SHOT_EXAMPLES + "\n\nRespond with valid JSON only."},
                {"role": "user", "content": f"Parse this query into JSON: {query}"}
            ],
            response_format={
                "type": "json_object"  # Use JSON mode (requires "json" in prompt)
            },
            temperature=0.1  # Low temperature for consistent parsing
        )

        # Extract the structured JSON response
        content = completion.choices[0].message.content
        if content:
            return json.loads(content)

        raise ValueError("No content in LLM response")

    def _parse_with_regex(self, query: str) -> Dict[str, Any]:
        """Fallback regex-based parser for simple queries."""
        result = {
            "intent": "fetch_emails",
            "people": [],
            "topics": [],
            "time_range": {},
            "limits": {"k": 50}
        }

        # Detect intent
        if any(word in query.lower() for word in ["summarize", "summary", "what did"]):
            result["intent"] = "summarize"
        elif any(word in query.lower() for word in ["receipt", "invoice", "order", "payment"]):
            result["intent"] = "list_receipts"
            result["limits"]["k"] = 100

        # Extract time ranges
        time_patterns = [
            (r"last (\d+) (hour|day|week|month|year)s?", lambda m: f"{m.group(1)} {m.group(2)}s"),
            (r"in the last (\d+) (hour|day|week|month|year)s?", lambda m: f"{m.group(1)} {m.group(2)}s"),
            (r"(yesterday|today)", lambda m: "1 day" if m.group(1) == "yesterday" else "0 days"),
            (r"this (week|month|year)", lambda m: f"1 {m.group(1)}"),
        ]

        for pattern, formatter in time_patterns:
            match = re.search(pattern, query.lower())
            if match:
                result["time_range"]["last_n"] = formatter(match)
                break

        # Extract people (simple: look for "from X")
        from_match = re.search(r"from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)", query)
        if from_match:
            result["people"].append(from_match.group(1))

        # Extract limit numbers
        limit_match = re.search(r"(?:last|latest|first)\s+(\d+)", query.lower())
        if limit_match:
            result["limits"]["k"] = int(limit_match.group(1))

        # Extract basic topics (words not already matched)
        topic_keywords = ["meeting", "project", "contract", "flight", "package", "password", "receipt", "invoice"]
        for keyword in topic_keywords:
            if keyword in query.lower():
                result["topics"].append(keyword)

        return result

    def _normalize_dates(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Convert relative dates to absolute timestamps."""
        time_range = query.get("time_range", {})

        # Handle last_n format
        if "last_n" in time_range:
            last_n = time_range["last_n"]
            now = datetime.utcnow()

            # Parse "3 days", "2 weeks", etc.
            match = re.match(r"(\d+)\s+(hour|day|week|month|quarter|year)s?", last_n)
            if match:
                amount = int(match.group(1))
                unit = match.group(2)

                if unit == "hour":
                    delta = timedelta(hours=amount)
                elif unit == "day":
                    delta = timedelta(days=amount)
                elif unit == "week":
                    delta = timedelta(weeks=amount)
                elif unit == "month":
                    delta = timedelta(days=amount * 30)  # Approximate
                elif unit == "quarter":
                    delta = timedelta(days=amount * 90)  # Approximate
                elif unit == "year":
                    delta = timedelta(days=amount * 365)  # Approximate
                else:
                    delta = timedelta(days=1)

                time_range["since"] = (now - delta).isoformat() + "Z"
                time_range["until"] = now.isoformat() + "Z"
                del time_range["last_n"]

        # Parse absolute dates in since/until
        for key in ["since", "until"]:
            if key in time_range and not time_range[key].endswith("Z"):
                try:
                    dt = date_parser.parse(time_range[key])
                    time_range[key] = dt.isoformat() + "Z"
                except:
                    pass

        query["time_range"] = time_range
        return query

    def _normalize_people(self, query: Dict[str, Any], db_connection) -> Dict[str, Any]:
        """Normalize people names against contacts database."""
        people = query.get("people", [])
        normalized = []

        for person in people:
            # Try to find in contacts table
            cursor = db_connection.cursor()

            # Try exact match first
            cursor.execute(
                "SELECT email, name FROM contacts WHERE LOWER(name) = LOWER(?) OR LOWER(email) = LOWER(?) LIMIT 1",
                (person, person)
            )
            result = cursor.fetchone()

            if result:
                normalized.append(result[0])  # Use email
            else:
                # Try fuzzy match on name
                cursor.execute(
                    "SELECT email, name FROM contacts WHERE LOWER(name) LIKE LOWER(?) LIMIT 1",
                    (f"%{person}%",)
                )
                result = cursor.fetchone()
                if result:
                    normalized.append(result[0])
                else:
                    # Keep original if no match
                    normalized.append(person)

        query["people"] = normalized
        return query


# Convenience function
def parse_query(query: str, db_connection=None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a natural language email query.

    Args:
        query: Natural language query string
        db_connection: Optional database connection for contact normalization
        api_key: Optional Anthropic API key (uses env var if not provided)

    Returns:
        Structured query dictionary

    Example:
        >>> parse_query("all updates from john in the last 3 days")
        {
            "intent": "fetch_emails",
            "people": ["john@example.com"],
            "time_range": {"since": "2025-10-27T...", "until": "2025-10-30T..."},
            "topics": [],
            "limits": {"k": 50}
        }
    """
    parser = QueryParser(api_key=api_key)
    return parser.parse(query, db_connection)
