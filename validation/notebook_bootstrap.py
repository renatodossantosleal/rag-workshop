"""Bootstrap comum para os notebooks _final do workshop.

Centraliza a configuracao repetida no inicio de cada notebook: definir
defaults de modelo/tag via variavel de ambiente, montar ARNs de modelo
Bedrock (foundation-model ou inference-profile) e instalar as tags
automaticas de recursos AWS criados durante a execucao.

Uso no notebook (primeira celula de codigo). A busca da raiz do projeto e o
ajuste do sys.path precisam ser feitos ANTES do import, pois e isso que
torna o pacote "validation" importavel a partir de qualquer subpasta:

    import sys
    from pathlib import Path
    ROOT = Path.cwd()
    while ROOT != ROOT.parent and not (ROOT / "validation").is_dir():
        ROOT = ROOT.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from validation.notebook_bootstrap import bootstrap
    bedrock_model_arn, load_workshop_state, persist_workshop_state_file = bootstrap()
"""

import os

import boto3


def _bedrock_model_arn(model_id, region_name=None):
    if model_id.startswith("arn:"):
        return model_id
    region_name = region_name or boto3.session.Session().region_name
    if model_id.startswith(("us.", "global.", "eu.", "apac.")):
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        return (
            f"arn:aws:bedrock:{region_name}:{account_id}:"
            f"inference-profile/{model_id}"
        )
    return f"arn:aws:bedrock:{region_name}::foundation-model/{model_id}"


def workshop_retrieved_references(response):
    """Extrai a lista de retrievedReferences da primeira citation de uma
    resposta de retrieve_and_generate (ou lista vazia se nao houver)."""
    citations = response.get("citations") or []
    if not citations:
        return []
    return citations[0].get("retrievedReferences") or []


def bootstrap():
    """Configura o ambiente do notebook e retorna os helpers necessarios.

    Retorna:
        bedrock_model_arn: funcao que monta o ARN correto (foundation-model
            ou inference-profile) para um model_id.
        load_workshop_state: funcao que le o estado compartilhado entre
            notebooks (validation/aws_state.json).
        persist_workshop_state_file: funcao que grava esse estado.
    """
    os.environ.setdefault("WORKSHOP_KB_TAG_VALUE", "true")
    os.environ.setdefault("RAG_WORKSHOP_TAG_VALUE", "true")
    os.environ.setdefault(
        "BEDROCK_TEXT_MODEL_ID",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    os.environ.setdefault(
        "BEDROCK_EVALUATION_MODEL_ID",
        "us.anthropic.claude-sonnet-4-6",
    )

    from validation.runtime_tags import install as _install_workshop_tags
    from validation.workshop_state import (
        load_workshop_state,
        persist_workshop_state as persist_workshop_state_file,
    )

    _install_workshop_tags()

    return _bedrock_model_arn, load_workshop_state, persist_workshop_state_file
