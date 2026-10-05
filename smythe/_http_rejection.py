"""A provider's refusal of a request before any model work, settled at zero cost.

A durable journal settles such a response as a known, free rejection, and so
does an ordinary run under a retry policy. ``smythe.workflow_store`` imports
these names from here. This module imports nothing from smythe.
"""

from __future__ import annotations

import re

_HTTP_REJECTION_SCOPE = "rejected_http_request"
# Statuses a provider returns before any model work: authentication,
# permission, unknown model or route, oversized request, and rate or quota
# limits. Each maps to the SDK status error that carries it; both SDKs raise
# APIStatusError for a status without its own class (OpenAI's 413). Every other
# status keeps unknown exposure: a 400 can follow billed generation, as with
# Anthropic's output content filter, and 408/499 are timeouts.
_REJECTION_STATUS_ERRORS = {
    401: "AuthenticationError", 403: "PermissionDeniedError", 404: "NotFoundError",
    413: "RequestTooLargeError", 429: "RateLimitError",
}
# The providers' plain error body: OpenAI's {"error": {"message", "type",
# "param", "code"}} and Anthropic's {"type": "error", "error": {"type",
# "message"}, "request_id": ...}. Another field, such as usage, may describe work.
_ERROR_BODY_FIELDS = frozenset({"error", "type", "request_id"})
_ERROR_OBJECT_FIELDS = frozenset({"message", "type", "param", "code"})
_ERROR_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def _http_rejection_receipt(evidence, raw, price_version):
    """Return a zero-cost receipt for a provider's refusal before model work.

    Only a 401, 403, 404, 413 or 429 response qualifies. Its body must be the
    provider's plain error object, whose fields are strings or null, with no
    usage or other field. The recorded exception must be that status's SDK
    error, ``APIStatusError`` or none; a transport error such as a timeout does
    not qualify. Anything else returns None: unknown exposure.
    """
    status = evidence["status_code"]
    if (type(status) is not int or status not in _REJECTION_STATUS_ERRORS
            or evidence["transport_error"] not in {None, "APIStatusError", _REJECTION_STATUS_ERRORS[status]}
            or type(raw) is not dict or not raw.keys() <= _ERROR_BODY_FIELDS
            or raw.get("type", "error") != "error" or type(raw.get("request_id", "")) is not str
            or type(raw.get("error")) is not dict or not raw["error"].keys() <= _ERROR_OBJECT_FIELDS
            or any(value is not None and type(value) is not str for value in raw["error"].values())):
        return None

    def identifier(value):
        return value if type(value) is str and _ERROR_IDENTIFIER.fullmatch(value) else None

    return {"version": 1, "price_version": price_version, "pricing_scope": _HTTP_REJECTION_SCOPE,
            "http_status": status, "error_type": identifier(raw["error"].get("type")),
            "error_code": identifier(raw["error"].get("code")),
            "cost_nanousd": 0, "cost_usd": "0", "cost_is_complete": True, "accounting_error": None}
