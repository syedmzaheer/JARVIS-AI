"""
RETRY UTILITY
=============

Calls a function and, if it raises, retries a few times with exponential backoff. 
Used for Groq and Tavily API calls so temporary rate limits or network blips 
don't immediately fail the request.

Example:
  response = with_retry (lambda: groq_client.chat(...), max_retries=3, initial_delay=1.0)
"""

import logging
import time
from typing import TypeVar, Callable


logger = logging.getLogger("J.A.R.V.I.S")

# Type variable: with_retry returns whatever the callable returns.
T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> T:
    """
    Execute fn(). If it raises, wait initial_delay seconds and try again; delay doubles each retry. 
    After max_retries attempts (including the first), re-raise the last exception.
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise