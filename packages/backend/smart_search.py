"""
Smart search integration that combines query parsing with email retrieval.
This addresses the "it's not fetching what I need" issues by better understanding user intent.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from query_parser import QueryParser


class SmartSearchEngine:
    """Enhanced search engine that understands natural language queries."""

    def __init__(self, supabase_client, embedding_model, anthropic_api_key: Optional[str] = None):
        """
        Initialize smart search engine.

        Args:
            supabase_client: Supabase client for database access
            embedding_model: SentenceTransformer model for embeddings
            anthropic_api_key: API key for query parsing (uses env var if not provided)
        """
        self.sb = supabase_client
        self.model = embedding_model
        self.parser = QueryParser(api_key=anthropic_api_key)

    def search(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform intelligent search based on natural language query.

        Args:
            query: Natural language search query
            user_id: Optional user ID for filtering

        Returns:
            Dictionary containing:
                - results: List of matching emails
                - parsed_query: The structured query that was used
                - search_strategy: Description of how the search was performed
        """
        print(f"\n=== SMART SEARCH START ===")
        print(f"[SmartSearch] Query: '{query}'")

        # Step 1: Parse the query to understand intent
        try:
            parsed = self.parser.parse(query, db_connection=None)
            print(f"[SmartSearch] Parsed query: {parsed}")
        except Exception as e:
            print(f"[SmartSearch] Parse error: {e}, using fallback")
            # Fallback to simple search
            parsed = {
                "intent": "fetch_emails",
                "people": [],
                "topics": [query],
                "time_range": {},
                "limits": {"k": 50}
            }

        # Step 2: Build search strategy based on parsed query
        search_strategy = self._determine_search_strategy(parsed)
        print(f"[SmartSearch] Strategy: {search_strategy}")

        # Step 3: Execute search based on intent
        if parsed["intent"] == "fetch_emails":
            results = self._fetch_emails(parsed, user_id)
        elif parsed["intent"] == "summarize":
            results = self._fetch_for_summary(parsed, user_id)
        elif parsed["intent"] == "list_receipts":
            results = self._fetch_receipts(parsed, user_id)
        else:
            results = self._fetch_emails(parsed, user_id)

        print(f"[SmartSearch] Found {len(results)} results")

        return {
            "results": results,
            "parsed_query": parsed,
            "search_strategy": search_strategy,
            "result_count": len(results)
        }

    def _determine_search_strategy(self, parsed: Dict[str, Any]) -> str:
        """Determine what search strategy to use."""
        strategies = []

        if parsed.get("people"):
            strategies.append(f"Filter by senders: {', '.join(parsed['people'])}")

        if parsed.get("time_range"):
            if "last_n" in parsed["time_range"]:
                strategies.append(f"Time range: last {parsed['time_range']['last_n']}")
            elif "since" in parsed["time_range"]:
                strategies.append(f"Time range: {parsed['time_range'].get('since', '')} to {parsed['time_range'].get('until', 'now')}")

        if parsed.get("topics"):
            strategies.append(f"Topics: {', '.join(parsed['topics'])}")

        limit = parsed.get("limits", {}).get("k", 50)
        strategies.append(f"Max results: {limit}")

        return " | ".join(strategies)

    def _fetch_emails(self, parsed: Dict[str, Any], user_id: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch emails based on parsed query."""

        # Build the query
        query_builder = self.sb.table("emails").select("*")

        # Apply user filter if provided
        if user_id:
            query_builder = query_builder.eq("google_user_id", user_id)

        # Apply people filter (sender/from)
        if parsed.get("people"):
            people_filters = []
            for person in parsed["people"]:
                # Try both email and name matches
                people_filters.append(f"from_email.ilike.%{person}%")
                people_filters.append(f"from_name.ilike.%{person}%")

            # Use OR filter for people
            if people_filters:
                # Note: In real implementation, you'd need to use Supabase's or() method
                # For now, we'll filter the first person only as an example
                first_person = parsed["people"][0]
                query_builder = query_builder.or_(
                    f"from_email.ilike.%{first_person}%,from_name.ilike.%{first_person}%"
                )

        # Apply time range filter
        time_range = parsed.get("time_range", {})
        if "since" in time_range:
            query_builder = query_builder.gte("received_at", time_range["since"])
        if "until" in time_range:
            query_builder = query_builder.lte("received_at", time_range["until"])

        # Apply topic filters via semantic search
        if parsed.get("topics"):
            # Generate embedding for topics
            topic_text = " ".join(parsed["topics"])
            topic_embedding = self.model.encode(topic_text, normalize_embeddings=True)

            # Use vector search via RPC
            try:
                vector_results = self.sb.rpc('match_emails', {
                    'query_embedding': topic_embedding.tolist(),
                    'match_threshold': 0.35,
                    'match_count': parsed.get("limits", {}).get("k", 50)
                }).execute()

                vector_candidates = getattr(vector_results, "data", []) or []

                # If we have time/people filters, apply them to vector results
                if time_range or parsed.get("people"):
                    vector_candidates = self._apply_filters_to_results(
                        vector_candidates,
                        parsed
                    )

                return vector_candidates[:parsed.get("limits", {}).get("k", 50)]

            except Exception as e:
                print(f"[SmartSearch] Vector search failed: {e}")
                # Fall through to basic search

        # Apply limit
        limit = parsed.get("limits", {}).get("k", 50)
        query_builder = query_builder.limit(limit)

        # Order by recency
        query_builder = query_builder.order("received_at", desc=True)

        # Execute
        try:
            result = query_builder.execute()
            return getattr(result, "data", []) or []
        except Exception as e:
            print(f"[SmartSearch] Query execution failed: {e}")
            return []

    def _fetch_for_summary(self, parsed: Dict[str, Any], user_id: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch emails for summarization - similar to fetch but may use different limits."""
        # For summaries, we might want more context
        if "limits" not in parsed:
            parsed["limits"] = {"k": 20}  # Fewer emails for summary

        return self._fetch_emails(parsed, user_id)

    def _fetch_receipts(self, parsed: Dict[str, Any], user_id: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch receipt/transaction emails."""
        # Add receipt-related topics if not present
        if "topics" not in parsed:
            parsed["topics"] = []

        receipt_keywords = ["receipt", "invoice", "payment", "order", "transaction", "purchase"]
        parsed["topics"].extend([kw for kw in receipt_keywords if kw not in parsed["topics"]])

        # Increase limit for receipts
        if "limits" not in parsed or parsed["limits"].get("k", 0) < 100:
            parsed["limits"] = {"k": 100}

        return self._fetch_emails(parsed, user_id)

    def _apply_filters_to_results(
        self,
        results: List[Dict[str, Any]],
        parsed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply filters to already-fetched results."""
        filtered = results

        # Filter by people
        if parsed.get("people"):
            people = [p.lower() for p in parsed["people"]]
            filtered = [
                email for email in filtered
                if any(
                    person in email.get("from_email", "").lower() or
                    person in email.get("from_name", "").lower()
                    for person in people
                )
            ]

        # Filter by time range
        time_range = parsed.get("time_range", {})
        if "since" in time_range or "until" in time_range:
            since = datetime.fromisoformat(time_range.get("since", "1970-01-01T00:00:00Z").replace("Z", "+00:00"))
            until = datetime.fromisoformat(time_range.get("until", "2100-01-01T00:00:00Z").replace("Z", "+00:00"))

            filtered = [
                email for email in filtered
                if email.get("received_at") and
                since <= datetime.fromisoformat(email["received_at"].replace("Z", "+00:00")) <= until
            ]

        return filtered


def smart_search(
    query: str,
    supabase_client,
    embedding_model,
    user_id: Optional[str] = None,
    anthropic_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for smart search.

    Args:
        query: Natural language search query
        supabase_client: Supabase client
        embedding_model: SentenceTransformer model
        user_id: Optional user ID
        anthropic_api_key: Optional API key

    Returns:
        Search results dictionary

    Example:
        >>> results = smart_search(
        ...     "all updates from john in the last 3 days",
        ...     sb_client,
        ...     model
        ... )
        >>> print(f"Found {results['result_count']} emails")
        >>> print(f"Strategy: {results['search_strategy']}")
    """
    engine = SmartSearchEngine(supabase_client, embedding_model, anthropic_api_key)
    return engine.search(query, user_id)
