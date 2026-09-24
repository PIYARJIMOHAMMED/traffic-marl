"""
Security utilities for the Traffic MARL backend.

Prototype-level authentication and role-based authorization.
For production deployment, replace the demo token store with an
external identity provider / secret manager.
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class User:
    username: str
    role: str


# Prototype credentials.
# Do NOT store real passwords/secrets in source code.
# Tokens should be supplied through environment variables in deployment.
DEMO_TOKENS = {
    "demo-admin-token": User("admin", "admin"),
    "demo-operator-token": User("operator", "operator"),
    "demo-viewer-token": User("viewer", "viewer"),
}


def authenticate(authorization: Optional[str]) -> User:
    """Authenticate a Bearer token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme",
        )

    token = authorization.split(" ", 1)[1].strip()

    user = DEMO_TOKENS.get(token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
        )

    return user


def require_role(user: User, allowed_roles: set[str]) -> User:
    """Authorize a user against an allowed-role set."""
    if user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

    return user


def get_current_user(
    authorization: Optional[str] = Header(default=None),
) -> User:
    return authenticate(authorization)


def require_operator(
    authorization: Optional[str] = Header(default=None),
) -> User:
    user = authenticate(authorization)
    return require_role(user, {"admin", "operator"})


def require_admin(
    authorization: Optional[str] = Header(default=None),
) -> User:
    user = authenticate(authorization)
    return require_role(user, {"admin"})