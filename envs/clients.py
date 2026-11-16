"""Separate API clients, bounded transient retries, no global OpenAI settings."""
import os
import threading
import time
from urllib.parse import urlsplit

from openai import OpenAI, APIConnectionError, APIStatusError


class EndpointError(RuntimeError):
    def __init__(self, source, kind, status=None):
        self.source, self.kind, self.status = source, kind, status
        # Avoid copying remote response bodies, URLs with credentials, or headers.
        super().__init__(f"{source}: {kind}, HTTP={status}")


class Endpoint:
    def __init__(self, prefix, concurrency, retries=2, timeout=120):
        self.source = prefix
        self.base_url = os.environ.get(prefix + "_BASE_URL", "")
        self.model = os.environ.get(prefix + "_MODEL", "")
        key = os.environ.get(prefix + "_API_KEY", "")
        if not all((self.base_url, self.model, key)):
            raise ValueError(f"Set {prefix}_BASE_URL / {prefix}_MODEL / {prefix}_API_KEY")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError(f"Invalid {prefix}_BASE_URL: use a credential-free http(s) URL")
        self.client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout, max_retries=0)
        self.limit = threading.BoundedSemaphore(concurrency)
        self.retries = retries

    def complete(self, messages, events, **kwargs):
        for attempt in range(self.retries + 1):
            start = time.monotonic()
            try:
                with self.limit:
                    response = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
                events.append({"source": self.source, "attempt": attempt, "elapsed_seconds": time.monotonic() - start,
                               "usage": response.usage.model_dump() if response.usage else None,
                               "response": response.model_dump()})
                return response
            except (APIConnectionError, APIStatusError) as error:
                status = getattr(error, "status_code", None)
                transient = status in (408, 409, 429) or (status is not None and status >= 500) or status is None
                # Context errors are classified from the body but the body is not logged.
                body = str(getattr(error, "body", "")).lower()
                kind = "context_overflow" if any(s in body for s in ("context length", "maximum context", "max_model_len", "context_length_exceeded")) else type(error).__name__
                events.append({"source": self.source, "attempt": attempt, "elapsed_seconds": time.monotonic() - start,
                               "error": kind, "http_status": status})
                if not transient or attempt == self.retries:
                    raise EndpointError(self.source, kind, status) from None
                time.sleep(min(2 ** attempt, 8))

    def public_config(self):
        # URL is hashed, to avoid publishing a private provider hostname/path.
        from eval.storage import digest
        return {"model": self.model, "endpoint_sha256": digest(self.base_url), "max_retries": self.retries}
