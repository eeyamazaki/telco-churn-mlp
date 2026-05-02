# region importações

import os
from datetime import datetime, timedelta

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# endregion

# Importação da .env
load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")

if not SECRET_KEY:
    raise ValueError("JWT_SECRET não definido no .env")

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", 30))

# Simulação de usuários de um banco de dados
USER_DB = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
}

# Chamando chave de criptografia
security = HTTPBearer()


# Classe do login
class LoginRequest(BaseModel):
    username: str
    password: str


def create_token(username: str, role: str) -> str:
    """
    Cria um token com usuário, permissão e expiração
    """
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    # Retorno final é um token codificado
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)) -> dict:  # noqa: B008
    """
    Faz a validação do token, para saber se ele é inválido ou expirado
    """
    # Validação do token
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])

        # Validação se realmente existe usuário e senha
        username = payload.get("sub")
        role = payload.get("role")

        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Token inválido")

        # Se tudo der certo, retorna usuário
        return {"username": payload["sub"], "role": payload["role"]}
    # Caso do token estar expirado
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=401, detail="Token expirado. Faça login novamente"
        ) from err
    # Caso do token estar inválido
    except jwt.InvalidTokenError as err:
        raise HTTPException(status_code=401, detail="Token Inválido") from err


def authenticate_user(username: str, password: str) -> dict | None:
    """
    Verificação de usuário e senha
    """

    # Usuário do "Banco"
    user = USER_DB.get(username)

    # Se tudo der certo, retorna dados do usuário
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None
