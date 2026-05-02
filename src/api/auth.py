"""Autenticação JWT para a API de previsão de churn."""

import os
from datetime import UTC, datetime, timedelta

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")

if not SECRET_KEY:
    raise ValueError("JWT_SECRET não definido no .env")

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", 30))

# Simulação de banco de usuários — substituir por integração real em produção
USER_DB = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
}

security = HTTPBearer()


class LoginRequest(BaseModel):
    username: str
    password: str


def create_token(username: str, role: str) -> str:
    """Cria um JWT com usuário, permissão e tempo de expiração."""
    expire = datetime.now(UTC) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)) -> dict:  # noqa: B008
    """Valida o token JWT e retorna os dados do usuário autenticado."""
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Token inválido")
        return {"username": username, "role": role}
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=401, detail="Token expirado. Faça login novamente"
        ) from err
    except jwt.InvalidTokenError as err:
        raise HTTPException(status_code=401, detail="Token inválido") from err


def authenticate_user(username: str, password: str) -> dict | None:
    """Verifica credenciais e retorna dados do usuário ou None."""
    user = USER_DB.get(username)
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None
