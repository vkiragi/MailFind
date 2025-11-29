"""
Adaptive Search Module
Implements quantile-based adaptive thresholds for semantic similarity search.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json


class AdaptiveSearchEngine:
    """
    Manages adaptive quantile-based threshold selection for semantic search.

    Key features:
    - Calculates quantiles (10th, 20th, 50th, 80th, 90th percentiles) from similarity scores
    - Stores and updates quantiles per query type in Supabase
    - Provides adaptive threshold selection based on user preferences
    - Logs search queries and metadata for analytics
    """

    def __init__(self, supabase_client):
        self.sb = supabase_client

        # Query type patterns
        self.query_patterns = {
            'temporal': ['recent', 'latest', 'newest', 'today', 'yesterday', 'this week', 'last week'],
            'news': ['news', 'nyt', 'new york times', 'newspaper', 'breaking'],
            'sports': ['fantasy', 'football', 'basketball', 'baseball', 'sports', 'nfl', 'nba', 'mlb'],
            'latest': ['latest', 'most recent', 'newest', 'last email'],
        }

    def classify_query_type(self, query: str) -> str:
        """
        Classify query into one of: temporal, news, sports, latest, default.

        Args:
            query: The user's natural language query

        Returns:
            Query type string
        """
        query_lower = query.lower()

        # Check each pattern - more specific patterns first
        if any(term in query_lower for term in self.query_patterns['sports']):
            return 'sports'
        elif any(term in query_lower for term in self.query_patterns['news']):
            return 'news'
        elif any(term in query_lower for term in self.query_patterns['latest']):
            return 'latest'
        elif any(term in query_lower for term in self.query_patterns['temporal']):
            return 'temporal'
        else:
            return 'default'

    def calculate_quantiles(self, similarity_scores: List[float]) -> Dict[str, float]:
        """
        Calculate quantile statistics from similarity scores.

        Args:
            similarity_scores: List of cosine similarity scores (0.0-1.0)

        Returns:
            Dictionary with percentile_10, percentile_20, percentile_50, percentile_80, percentile_90
        """
        if not similarity_scores:
            return {
                'percentile_10': 0.0,
                'percentile_20': 0.0,
                'percentile_50': 0.0,
                'percentile_80': 0.0,
                'percentile_90': 0.0,
            }

        scores_array = np.array(similarity_scores)

        return {
            'percentile_10': float(np.percentile(scores_array, 10)),
            'percentile_20': float(np.percentile(scores_array, 20)),
            'percentile_50': float(np.percentile(scores_array, 50)),
            'percentile_80': float(np.percentile(scores_array, 80)),
            'percentile_90': float(np.percentile(scores_array, 90)),
        }

    def get_quantile_threshold(self, query_type: str, percentile: int = 80, google_user_id: Optional[str] = None) -> float:
        """
        Get adaptive threshold for a query type based on stored quantiles.

        Args:
            query_type: One of: temporal, news, sports, latest, default
            percentile: Which percentile to use (10, 20, 50, 80, 90). Default: 80
            google_user_id: Optional user ID for personalized thresholds

        Returns:
            Threshold value (0.0-1.0)
        """
        try:
            # First, check if user has custom preferences
            if google_user_id:
                prefs_result = self.sb.table("user_search_preferences").select("*").eq("google_user_id", google_user_id).limit(1).execute()
                prefs_data = getattr(prefs_result, "data", []) or []

                if prefs_data:
                    user_prefs = prefs_data[0]
                    # Use user's preferred quantile if set
                    if user_prefs.get('preferred_quantile'):
                        percentile = int(user_prefs['preferred_quantile'] * 100)

                    # Apply custom threshold offset if set
                    custom_offset = user_prefs.get('custom_threshold_offset', 0.0)

            # Get quantile data for this query type
            result = self.sb.table("query_type_quantiles").select("*").eq("query_type", query_type).limit(1).execute()
            data = getattr(result, "data", []) or []

            if not data:
                print(f"[AdaptiveSearch] No quantile data found for query_type '{query_type}', using defaults")
                return self._get_default_threshold(query_type)

            quantile_row = data[0]

            # Get the appropriate percentile column
            percentile_key = f"percentile_{percentile}"
            threshold = quantile_row.get(percentile_key)

            if threshold is None or threshold == 0.0:
                print(f"[AdaptiveSearch] No {percentile_key} data for '{query_type}', using default")
                return self._get_default_threshold(query_type)

            # Apply user's custom offset if they have one
            if google_user_id and 'custom_offset' in locals():
                threshold += custom_offset
                threshold = max(0.0, min(1.0, threshold))  # Clamp to [0.0, 1.0]

            print(f"[AdaptiveSearch] Using adaptive threshold {threshold:.4f} for '{query_type}' (p{percentile})")
            return float(threshold)

        except Exception as e:
            print(f"[AdaptiveSearch] Error getting quantile threshold: {e}")
            return self._get_default_threshold(query_type)

    def _get_default_threshold(self, query_type: str) -> float:
        """Fallback thresholds if quantile data unavailable."""
        defaults = {
            'temporal': 0.15,
            'news': 0.20,
            'sports': 0.25,
            'latest': 0.20,
            'default': 0.30,
        }
        return defaults.get(query_type, 0.30)

    def update_quantiles(self, query_type: str, similarity_scores: List[float]) -> None:
        """
        Update quantile statistics for a query type using exponential moving average.

        This implements a rolling update to adapt to changing data distributions over time.

        Args:
            query_type: The query type to update
            similarity_scores: List of similarity scores from the latest search
        """
        try:
            if not similarity_scores:
                print(f"[AdaptiveSearch] No similarity scores to update quantiles for '{query_type}'")
                return

            # Calculate new quantiles from current batch
            new_quantiles = self.calculate_quantiles(similarity_scores)

            # Get existing quantiles
            result = self.sb.table("query_type_quantiles").select("*").eq("query_type", query_type).limit(1).execute()
            data = getattr(result, "data", []) or []

            if not data:
                # First time - insert new record
                self.sb.table("query_type_quantiles").insert({
                    'query_type': query_type,
                    'percentile_10': new_quantiles['percentile_10'],
                    'percentile_20': new_quantiles['percentile_20'],
                    'percentile_50': new_quantiles['percentile_50'],
                    'percentile_80': new_quantiles['percentile_80'],
                    'percentile_90': new_quantiles['percentile_90'],
                    'sample_count': len(similarity_scores),
                    'last_updated': datetime.now().isoformat(),
                }).execute()
                print(f"[AdaptiveSearch] Initialized quantiles for '{query_type}' with {len(similarity_scores)} samples")
            else:
                # Update using exponential moving average (alpha=0.1 means 10% new, 90% old)
                existing = data[0]
                alpha = 0.1  # Smoothing factor - lower = slower adaptation
                sample_count = existing.get('sample_count', 0) + len(similarity_scores)

                updated_quantiles = {
                    'percentile_10': alpha * new_quantiles['percentile_10'] + (1 - alpha) * existing.get('percentile_10', 0),
                    'percentile_20': alpha * new_quantiles['percentile_20'] + (1 - alpha) * existing.get('percentile_20', 0),
                    'percentile_50': alpha * new_quantiles['percentile_50'] + (1 - alpha) * existing.get('percentile_50', 0),
                    'percentile_80': alpha * new_quantiles['percentile_80'] + (1 - alpha) * existing.get('percentile_80', 0),
                    'percentile_90': alpha * new_quantiles['percentile_90'] + (1 - alpha) * existing.get('percentile_90', 0),
                    'sample_count': sample_count,
                    'last_updated': datetime.now().isoformat(),
                }

                self.sb.table("query_type_quantiles").update(updated_quantiles).eq("query_type", query_type).execute()
                print(f"[AdaptiveSearch] Updated quantiles for '{query_type}' (total samples: {sample_count})")
                print(f"[AdaptiveSearch] New p80: {updated_quantiles['percentile_80']:.4f}")

        except Exception as e:
            print(f"[AdaptiveSearch] Error updating quantiles: {e}")

    def _get_user_id_from_google_id(self, google_user_id: str) -> Optional[str]:
        """
        Get user's UUID (id) from google_user_id.

        Args:
            google_user_id: Google OAuth user ID

        Returns:
            User's UUID if found, None otherwise
        """
        try:
            result = self.sb.table("users").select("id").eq("google_user_id", google_user_id).limit(1).execute()
            data = getattr(result, "data", []) or []

            if data:
                return data[0].get('id')
            return None
        except Exception as e:
            print(f"[AdaptiveSearch] Error getting user_id: {e}")
            return None

    def log_search_query(
        self,
        query_text: str,
        query_type: str,
        google_user_id: Optional[str],
        results_count: int,
        threshold_used: float,
        percentile_used: float,
        similarity_scores: List[float]
    ) -> Optional[str]:
        """
        Log a search query to the search_queries table for analytics.

        Args:
            query_text: The user's query
            query_type: Classified query type
            google_user_id: User ID if available
            results_count: Number of results returned
            threshold_used: Threshold value used
            percentile_used: Percentile value used (e.g., 0.80)
            similarity_scores: All similarity scores from the search

        Returns:
            search_query_id if successful, None otherwise
        """
        try:
            # Calculate statistics
            avg_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0
            max_similarity = float(np.max(similarity_scores)) if similarity_scores else 0.0
            min_similarity = float(np.min(similarity_scores)) if similarity_scores else 0.0

            # Get user_id (UUID) from google_user_id for v2 schema
            user_id = None
            if google_user_id:
                user_id = self._get_user_id_from_google_id(google_user_id)

            # Insert with both user_id (FK) and google_user_id (denormalized)
            insert_data = {
                'google_user_id': google_user_id,
                'query_text': query_text,
                'query_type': query_type,
                'results_count': results_count,
                'threshold_used': threshold_used,
                'percentile_used': percentile_used,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'min_similarity': min_similarity,
                'created_at': datetime.now().isoformat(),
            }

            # Add user_id if found (v2 schema compatibility)
            if user_id:
                insert_data['user_id'] = user_id

            result = self.sb.table("search_queries").insert(insert_data).execute()

            data = getattr(result, "data", []) or []
            if data:
                search_query_id = data[0].get('id')
                print(f"[AdaptiveSearch] Logged search query: '{query_text[:50]}...' (id: {search_query_id})")
                return search_query_id

            return None

        except Exception as e:
            print(f"[AdaptiveSearch] Error logging search query: {e}")
            return None

    def get_user_preferences(self, google_user_id: str) -> Dict:
        """
        Get user's search preferences or create default if not exists.

        Args:
            google_user_id: User's Google ID

        Returns:
            Dictionary with user preferences
        """
        try:
            result = self.sb.table("user_search_preferences").select("*").eq("google_user_id", google_user_id).limit(1).execute()
            data = getattr(result, "data", []) or []

            if data:
                return data[0]
            else:
                # Create default preferences with user_id lookup for v2 schema
                user_id = self._get_user_id_from_google_id(google_user_id)

                default_prefs = {
                    'google_user_id': google_user_id,
                    'precision_level': 'balanced',
                    'custom_threshold_offset': 0.0,
                    'preferred_quantile': 0.80,
                    'feature_weights': None,
                    'total_searches': 0,
                    'total_clicks': 0,
                    'avg_ctr': 0.0,
                }

                # Add user_id if found (v2 schema compatibility)
                if user_id:
                    default_prefs['user_id'] = user_id

                insert_result = self.sb.table("user_search_preferences").insert(default_prefs).execute()
                insert_data = getattr(insert_result, "data", []) or []

                if insert_data:
                    print(f"[AdaptiveSearch] Created default preferences for user {google_user_id}")
                    return insert_data[0]

                return default_prefs

        except Exception as e:
            print(f"[AdaptiveSearch] Error getting user preferences: {e}")
            return {
                'precision_level': 'balanced',
                'preferred_quantile': 0.80,
                'custom_threshold_offset': 0.0,
            }

    def increment_user_search_count(self, google_user_id: str) -> None:
        """Increment the user's total search count."""
        try:
            # Get current count
            prefs = self.get_user_preferences(google_user_id)
            total_searches = prefs.get('total_searches', 0) + 1

            self.sb.table("user_search_preferences").update({
                'total_searches': total_searches,
                'last_updated': datetime.now().isoformat(),
            }).eq("google_user_id", google_user_id).execute()

        except Exception as e:
            print(f"[AdaptiveSearch] Error incrementing search count: {e}")


def perform_adaptive_search(
    supabase_client,
    query_embedding: List[float],
    query_text: str,
    google_user_id: Optional[str] = None,
    max_results: int = 30,
    preferred_percentile: int = 80
) -> Tuple[List[Dict], Dict]:
    """
    Perform semantic search with adaptive quantile-based threshold.

    This is the main entry point for adaptive search. It:
    1. Classifies the query type
    2. Gets adaptive threshold based on quantiles
    3. Performs the search
    4. Logs the query
    5. Updates quantiles for future searches

    Args:
        supabase_client: Supabase client instance
        query_embedding: Query embedding vector
        query_text: User's original query text
        google_user_id: User ID if available
        max_results: Maximum number of results to return
        preferred_percentile: Percentile to use for threshold (10, 20, 50, 80, 90)

    Returns:
        Tuple of (search_results, metadata)
        metadata includes: query_type, threshold_used, percentile_used, search_query_id
    """
    engine = AdaptiveSearchEngine(supabase_client)

    # 1. Classify query type
    query_type = engine.classify_query_type(query_text)
    print(f"[AdaptiveSearch] Query classified as: {query_type}")

    # 2. Get adaptive threshold
    threshold = engine.get_quantile_threshold(query_type, preferred_percentile, google_user_id)

    # 3. Perform search with adaptive threshold
    try:
        results = supabase_client.rpc('match_emails', {
            'query_embedding': query_embedding,
            'match_threshold': threshold,
            'match_count': max_results
        }).execute()

        search_results = getattr(results, "data", []) or []

        # Extract similarity scores from results
        similarity_scores = [float(r.get('similarity', 0.0)) for r in search_results]

        print(f"[AdaptiveSearch] Found {len(search_results)} results with threshold {threshold:.4f}")

        # 4. Log the search query
        search_query_id = engine.log_search_query(
            query_text=query_text,
            query_type=query_type,
            google_user_id=google_user_id,
            results_count=len(search_results),
            threshold_used=threshold,
            percentile_used=preferred_percentile / 100.0,
            similarity_scores=similarity_scores
        )

        # 5. Update quantiles with new data (async/background in production)
        engine.update_quantiles(query_type, similarity_scores)

        # 6. Increment user search count
        if google_user_id:
            engine.increment_user_search_count(google_user_id)

        metadata = {
            'query_type': query_type,
            'threshold_used': threshold,
            'percentile_used': preferred_percentile,
            'search_query_id': search_query_id,
            'results_count': len(search_results),
        }

        return search_results, metadata

    except Exception as e:
        print(f"[AdaptiveSearch] Error performing adaptive search: {e}")
        raise
