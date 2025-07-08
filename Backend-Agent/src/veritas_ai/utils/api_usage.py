import json
import time
from datetime import datetime, date
from typing import Dict, Any, List

class APIUsageError(Exception):
    """Custom exception for API usage limit errors."""
    pass

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []

    def check(self) -> bool:
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            return False
        return True

    def add_call(self):
        self.calls.append(time.time())

class APIUsageManager:
    def __init__(self, filepath: str = 'api_usage.json'):
        self.filepath = filepath
        self.usage_data = self._load_usage_data()
        self.gemini_rate_limiter = RateLimiter(max_calls=14, period=60)

    def _load_usage_data(self) -> Dict[str, Any]:
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                # Ensure all keys are present
                data.setdefault('apify', {'count': 0})
                data.setdefault('tavily', {'count': 0})
                data.setdefault('gemini', {'count': 0, 'last_reset': str(date.today())})
                return data
        except FileNotFoundError:
            return self._default_usage_data()

    def _default_usage_data(self) -> Dict[str, Any]:
        return {
            'apify': {'count': 0},
            'tavily': {'count': 0},
            'gemini': {'count': 0, 'last_reset': str(date.today())}
        }

    def _save_usage_data(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.usage_data, f, indent=4)

    def _reset_daily_counters_if_needed(self):
        today = str(date.today())
        if self.usage_data['gemini']['last_reset'] != today:
            self.usage_data['gemini']['count'] = 0
            self.usage_data['gemini']['last_reset'] = today
            self._save_usage_data()

    def check_and_increment_apify(self):
        if self.usage_data['apify']['count'] >= 600:
            raise APIUsageError("Apify API call limit of 800 reached.")
        self.usage_data['apify']['count'] += 1
        self._save_usage_data()

    def check_and_increment_tavily(self):
        if self.usage_data['tavily']['count'] >= 1000:
            raise APIUsageError("Tavily API call limit of 1000 reached.")
        self.usage_data['tavily']['count'] += 1
        self._save_usage_data()

    def check_and_increment_gemini(self):
        self._reset_daily_counters_if_needed()
        
        # Check daily limit
        if self.usage_data['gemini']['count'] >= 800:
            raise APIUsageError("Gemini API daily call limit of 800 reached.")
        
        # Check rate limit
        if not self.gemini_rate_limiter.check():
            raise APIUsageError("Gemini API rate limit of 14 calls per minute exceeded.")
            
        self.usage_data['gemini']['count'] += 1
        self.gemini_rate_limiter.add_call()
        self._save_usage_data()

    def get_limits_status(self) -> Dict[str, bool]:
        self._reset_daily_counters_if_needed()
        return {
            "apify_limit_reached": self.usage_data['apify']['count'] >= 800,
            "tavily_limit_reached": self.usage_data['tavily']['count'] >= 1000,
            "gemini_daily_limit_reached": self.usage_data['gemini']['count'] >= 800,
        }

# Singleton instance
api_usage_manager = APIUsageManager() 