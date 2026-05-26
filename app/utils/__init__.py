"""
UTILITIES PACKAGE
=================

Helpers used by the services (no HTTP, no business logic)

  time_info            - get_time_information() : returns a string wuth current date/time fir the LLM prompt.
  retry                - with_retry(fn):  calls fn(); on failure retries with expotential backoff (Groq or Tavily).
"""