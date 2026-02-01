"""
Input validation module for the Hawkins Truth Engine.

Provides validation functions for API inputs to prevent DoS, memory exhaustion,
and other security issues.
"""

from __future__ import annotations

from urllib.parse import urlparse

from .config import MAX_CONTENT_LENGTH, MAX_URL_LENGTH, MIN_CONTENT_LENGTH


class ValidationError(ValueError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(message)


def validate_content_length(content: str, input_type: str) -> None:
    """
    Validate content length is within acceptable bounds.
    
    Args:
        content: The content string to validate
        input_type: Type of input ('raw_text', 'url', 'social_post')
        
    Raises:
        ValidationError: If content length is invalid
    """
    if content is None:
        raise ValidationError("Content cannot be null", field="content")
    
    if input_type == "url":
        if len(content) > MAX_URL_LENGTH:
            raise ValidationError(
                f"URL exceeds maximum length of {MAX_URL_LENGTH} characters",
                field="content"
            )
        if len(content) < 10:  # Minimum valid URL length (e.g., "http://a.b")
            raise ValidationError(
                "URL is too short to be valid",
                field="content"
            )
    else:
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValidationError(
                f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} bytes "
                f"(received {len(content)} bytes)",
                field="content"
            )
        if len(content.strip()) < MIN_CONTENT_LENGTH:
            raise ValidationError(
                f"Content is too short (minimum {MIN_CONTENT_LENGTH} characters required)",
                field="content"
            )


def validate_url_format(url: str) -> None:
    """
    Validate URL format is acceptable.
    
    Args:
        url: The URL string to validate
        
    Raises:
        ValidationError: If URL format is invalid
    """
    try:
        parsed = urlparse(url)
        
        # Must have a scheme
        if not parsed.scheme:
            raise ValidationError(
                "URL must include a scheme (http:// or https://)",
                field="content"
            )
        
        # Scheme must be http or https
        if parsed.scheme not in ("http", "https"):
            raise ValidationError(
                f"URL scheme '{parsed.scheme}' is not supported. Use http or https.",
                field="content"
            )
        
        # Must have a netloc (domain)
        if not parsed.netloc:
            raise ValidationError(
                "URL must include a valid domain",
                field="content"
            )
            
    except ValueError as e:
        raise ValidationError(f"Invalid URL format: {e}", field="content")


def validate_input_type(input_type: str) -> None:
    """
    Validate input type is one of the accepted values.
    
    Args:
        input_type: The input type to validate
        
    Raises:
        ValidationError: If input type is invalid
    """
    valid_types = {"raw_text", "url", "social_post"}
    
    if input_type not in valid_types:
        raise ValidationError(
            f"Invalid input_type '{input_type}'. Must be one of: {', '.join(sorted(valid_types))}",
            field="input_type"
        )


def validate_analyze_request(req) -> None:
    """
    Validate an AnalyzeRequest before processing.
    
    Args:
        req: The AnalyzeRequest object to validate
        
    Raises:
        ValidationError: If request validation fails
    """
    # Validate input type
    validate_input_type(req.input_type)
    
    # Validate content length
    validate_content_length(req.content, req.input_type)
    
    # Additional URL-specific validation
    if req.input_type == "url":
        validate_url_format(req.content)
