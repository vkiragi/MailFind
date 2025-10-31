"""
Test the query parser with various examples.
Run: python test_query_parser.py
"""

import json
from query_parser import parse_query

# Test queries
TEST_QUERIES = [
    "all updates from john in the last 3 days",
    "show me receipts from DoorDash this month",
    "summarize all emails about the contract negotiation",
    "flight confirmations from United Airlines",
    "emails from sarah about the project in the last week",
    "all invoices over $500 since January",
    "meeting notes from yesterday",
    "show me the last 10 emails from my boss",
    "package delivery notifications this week",
    "what did alice say about the bug fix?",
    "latest 5 newsletters from TechCrunch",
    "password reset emails from the last 2 hours",
]


def test_parser():
    """Test the parser with sample queries."""
    print("=" * 80)
    print("QUERY PARSER TEST")
    print("=" * 80)
    print()

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{i}. Query: \"{query}\"")
        print("-" * 80)

        try:
            result = parse_query(query)
            print("Parsed structure:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")

        print()


def test_with_db():
    """Test parser with database connection for name normalization."""
    import sqlite3

    print("\n" + "=" * 80)
    print("TESTING WITH DATABASE NORMALIZATION")
    print("=" * 80)

    # Create a test database with some contacts
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE contacts (
            email TEXT PRIMARY KEY,
            name TEXT,
            domain TEXT
        )
    """)

    # Insert test contacts
    test_contacts = [
        ("john@company.com", "John Smith", "company.com"),
        ("sarah@company.com", "Sarah Johnson", "company.com"),
        ("alice@tech.com", "Alice Brown", "tech.com"),
        ("noreply@doordash.com", "DoorDash", "doordash.com"),
        ("support@united.com", "United Airlines", "united.com"),
    ]

    cursor.executemany(
        "INSERT INTO contacts (email, name, domain) VALUES (?, ?, ?)",
        test_contacts
    )
    conn.commit()

    # Test queries with name normalization
    test_queries_db = [
        "all updates from john in the last 3 days",
        "emails from sarah about the project",
        "show me receipts from DoorDash",
    ]

    for query in test_queries_db:
        print(f"\nQuery: \"{query}\"")
        print("-" * 80)

        result = parse_query(query, db_connection=conn)
        print("Normalized people:", result.get("people"))
        print()

    conn.close()


if __name__ == "__main__":
    print("Testing query parser...")
    print("\nNOTE: This requires ANTHROPIC_API_KEY in your environment.")
    print("If not set, it will fall back to regex parsing.\n")

    test_parser()
    test_with_db()

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
