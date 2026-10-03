"""Errors that map onto HTTP responses.

Each carries the status code and error type the API layer should
report, so the rest of the code can raise them without knowing
about HTTP.
"""

from typing import Any


class RouterError(Exception):
    status_code = 500
    error_type = "api_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "type": self.error_type}


class ConfigError(RouterError):
    error_type = "configuration_error"


class InvalidRequestError(RouterError):
    status_code = 400
    error_type = "invalid_request_error"


class NoModelsError(RouterError):
    status_code = 503
    error_type = "no_models_available"


class UpstreamError(RouterError):
    status_code = 502
    error_type = "upstream_error"
