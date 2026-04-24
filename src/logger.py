"""Configuração de logging estruturado usando structlog.

Este módulo deve ser inicializado UMA ÚNICA VEZ no ponto de entrada da aplicação
(script de treino, lifespan do FastAPI ou primeira célula do notebook) através
da função setup_logging(). Todos os outros módulos apenas chamam get_logger(__name__).

Modos disponíveis:
    - json_logs=False (padrão): saída colorida e legível para desenvolvimento.
    - json_logs=True: saída em JSON para produção e ambientes de nuvem.

Níveis de log disponíveis (do menos para o mais grave):
    debug → info → warning → error → critical

Exemplo de uso:

    Ponto de entrada (uma vez):
        from src.logger import setup_logging
        setup_logging(level="INFO", json_logs=False)

    Qualquer módulo:
        from src.logger import get_logger
        logger = get_logger(__name__)
        logger.info("dados carregados", linhas=7043, colunas=21)
        logger.warning("alta taxa de nulos", coluna="TotalCharges", pct=0.16)
        logger.error("arquivo não encontrado", caminho="telco.csv")
"""

import logging
import sys

import structlog

_CONFIGURED = False


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configura o structlog globalmente. Deve ser chamada uma única vez.

    Caso seja chamada mais de uma vez, a segunda chamada em diante é ignorada.
    Isso garante que múltiplos módulos chamando get_logger() não reconfiguram
    o sistema de logging.

    Args:
        level: Nível mínimo de log a ser emitido. Valores aceitos (case-insensitive):
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
            Valores inválidos recaem silenciosamente para "INFO".
        json_logs: Se True, emite logs em formato JSON (recomendado para produção
            e ambientes de nuvem). Se False, emite formato legível com cores
            (recomendado para desenvolvimento).
    """
    
    global _CONFIGURED
    if _CONFIGURED:
        return
    
    render = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            render,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    
    _CONFIGURED = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Retorna um logger nomeado pronto para uso.

    O parâmetro name é usado para identificar a origem do log em cada mensagem
    emitida. Utilize sempre __name__ para que o módulo de origem apareça
    automaticamente nos logs.

    Atenção: setup_logging() deve ter sido chamada antes no ponto de entrada
    da aplicação. Caso contrário, o structlog usará sua configuração padrão.

    Args:
        name: Nome do logger, geralmente o __name__ do módulo chamador.

    Returns:
        Logger estruturado pronto para uso com os métodos debug(), info(),
        warning(), error() e critical().

    Exemplo:
        logger = get_logger(__name__)
        logger.info("treinamento iniciado", epoch=1, lr=0.001)
    """
    
    return structlog.get_logger(name)
