import os

# Define JWT_SECRET para testes caso .env não exista (ex: CI ou clone novo)
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-pytest")
