"""Central configuration: endpoints, deployment names and Entra token helpers.

Resolution order for every setting: real environment variable > `.env` in the repo root > default.
Resource names derive from SUFFIX (default: 'vg' + first 6 chars of the subscription id), matching infra/config.sh.
Run `python3 -m common.config --check` to print the resolved values (secrets masked) and list missing settings.
"""
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from dotenv.parser import parse_stream

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path(os.getenv("ENV_FILE") or ROOT / ".env")
if ENV_PATH.exists():
    with ENV_PATH.open() as stream:
        for binding in parse_stream(stream):
            if binding.error:
                raise ValueError(f"Invalid dotenv syntax at line {binding.original.line} in {ENV_PATH}")
load_dotenv(ENV_PATH, override=False)  # real environment variables win


@lru_cache(maxsize=1)
def subscription_id() -> str:
    sub, legacy = os.getenv("SUBSCRIPTION_ID"), os.getenv("SUB")
    if sub and legacy and sub != legacy:
        raise ValueError("SUB and SUBSCRIPTION_ID conflict; set one subscription consistently")
    return sub or legacy or subprocess.check_output(
        ["az", "account", "show", "--query", "id", "-o", "tsv"], text=True).strip()


def env(name: str, default: str = "") -> str:
    return os.getenv(name) or default


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


SEARCH_API_VERSION = env("SEARCH_API_VERSION", "2026-08-01-preview")
DI_API_VERSION = env("DI_API_VERSION", "2024-11-30")
OPENAI_API_VERSION = env("OPENAI_API_VERSION", "2024-10-21")


@lru_cache(maxsize=1)
def _suffix() -> str:
    if os.getenv("SUFFIX"):
        return os.environ["SUFFIX"]
    return "vg" + subscription_id()[:6]


SUFFIX = _suffix()
RG = env("RG", "rg-semantic-rag-demo")
LOC_SEARCH = env("LOC_SEARCH", "centralindia")
LOC_FOUNDRY = env("LOC_FOUNDRY", "southindia")
SEARCH_NAME = env("SEARCH_NAME", f"srag-{SUFFIX}")
DI_NAME = env("DI_NAME", f"srag-di-{SUFFIX}")
AI_NAME = env("AI_NAME", f"srag-ai-{SUFFIX}")
STORAGE_ACCOUNT = env("STORAGE_ACCOUNT") or env("STORAGE_NAME", f"sragst{SUFFIX}")
SEARCH_ENDPOINT = env("SEARCH_ENDPOINT", f"https://{SEARCH_NAME}.search.windows.net")
DI_ENDPOINT = env("DI_ENDPOINT", f"https://{DI_NAME}.cognitiveservices.azure.com")
AI_ENDPOINT = env("AI_ENDPOINT", f"https://{AI_NAME}.cognitiveservices.azure.com")

CHAT_DEPLOYMENT = env("CHAT_DEPLOYMENT", "gpt-5.6-sol")
CHAT_MODEL_NAME = env("CHAT_MODEL_NAME", CHAT_DEPLOYMENT)  # underlying model when it differs from the deployment name
EMBED_DEPLOYMENT = env("EMBED_DEPLOYMENT", "text-embedding-3-large")
EMBED_MODEL_NAME = env("EMBED_MODEL_NAME", EMBED_DEPLOYMENT)
EMBED_DIMENSIONS = env_int("EMBED_DIMENSIONS", 3072)  # changing this needs a new index and a re-ingest

# Search objects, agents and blob containers
INDEX_NAME = env("INDEX_NAME", "dossier-chunks")
KS_NAME = env("KS_NAME", "dossier-ks")
KS_ALL_NAME = env("KS_ALL_NAME", "dossier-ks-all")
KB_NAME = env("KB_NAME", "dossier-kb")
KB_TRACE_NAME = env("KB_TRACE_NAME", "dossier-kb-trace")
AGENT_NAME = env("AGENT_NAME", "dossier-agent")
AGENT_TRACE_NAME = env("AGENT_TRACE_NAME", "dossier-agent-trace")
DOCS_CONTAINER = env("DOCS_CONTAINER", "docs")
FIGURES_CONTAINER = env("FIGURES_CONTAINER", "figures")

# Project provisioning must verify Agent Service readiness, not just ARM creation.
PROJECT_ACCOUNT = env("PROJECT_ACCOUNT")
PROJECT_NAME = env("PROJECT_NAME")
PROJECT_ENDPOINT = env("PROJECT_ENDPOINT") or (
    f"https://{PROJECT_ACCOUNT}.services.ai.azure.com/api/projects/{PROJECT_NAME}"
    if PROJECT_ACCOUNT and PROJECT_NAME else "")

# SharePoint document source (ingest/sp_sync.py)
SOURCE_MODE = env("SOURCE_MODE", "sharepoint")  # "sharepoint" or "local" (use ./docs only)
SP_TENANT = env("SP_TENANT")
SP_CLIENT_ID = env("SP_CLIENT_ID")
SP_CLIENT_SECRET = env("SP_CLIENT_SECRET")
SP_SITE_URL = env("SP_SITE_URL")
SP_DRIVE_ID = env("SP_DRIVE_ID")
SP_FOLDER = env("SP_FOLDER")
SP_FILE_PATTERN = env("SP_FILE_PATTERN", "*.pdf")


def require(*names: str) -> None:
    """Fail once with every missing setting listed (names are attributes of this module)."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise RuntimeError("missing configuration: " + ", ".join(missing) +
                           " (set them in .env or the environment; see .env.example)")


def project_endpoint() -> str:
    require("PROJECT_ENDPOINT")
    return PROJECT_ENDPOINT


def get_token(resource: str) -> str:
    """Entra access token for the signed-in `az` user (no API keys are used anywhere)."""
    return subprocess.check_output(
        ["az", "account", "get-access-token", "--resource", resource, "--query", "accessToken", "-o", "tsv"],
        text=True,
    ).strip()


def search_headers() -> dict:
    return {"Authorization": f"Bearer {get_token('https://search.azure.com')}", "Content-Type": "application/json"}


def cognitive_headers() -> dict:
    return {"Authorization": f"Bearer {get_token('https://cognitiveservices.azure.com')}", "Content-Type": "application/json"}


def _check(stage="application") -> int:
    required = ["SEARCH_ENDPOINT", "DI_ENDPOINT", "AI_ENDPOINT", "STORAGE_ACCOUNT"]
    if stage in ("models", "application"):
        required += ["CHAT_DEPLOYMENT", "EMBED_DEPLOYMENT"]
    if stage in ("project", "application"):
        required += ["PROJECT_ENDPOINT"]
    for k, v in sorted(globals().items()):
        if k.isupper() and not k.startswith("_") and k != "ROOT" and isinstance(v, (str, int)):
            shown = "***" if v and "SECRET" in k else v
            print(f"{k:22} {shown if shown != '' else '(unset)'}")
    missing = [k for k in required if not globals().get(k)]
    if stage == "application" and SOURCE_MODE == "sharepoint" and not (ROOT / ".secrets" / "sharepoint.json").exists():
        sp = [k for k in ("SP_TENANT", "SP_CLIENT_ID", "SP_CLIENT_SECRET", "SP_FOLDER") if not globals()[k]]
        if not (SP_SITE_URL or SP_DRIVE_ID):
            sp.append("SP_SITE_URL or SP_DRIVE_ID")
        missing += [f"{k} (needed for SOURCE_MODE=sharepoint)" for k in sp]
    print("\nmissing:", ", ".join(missing) if missing else "none")
    return 1 if missing else 0


def _shell():
    """Line-based exports consumed as data, never evaluated as shell code."""
    values = {k: globals()[k] for k in (
        "RG", "SUFFIX", "LOC_SEARCH", "LOC_FOUNDRY", "SEARCH_NAME", "DI_NAME", "AI_NAME",
        "DOCS_CONTAINER", "FIGURES_CONTAINER")}
    values.update(SUB=subscription_id(), SUBSCRIPTION_ID=subscription_id(),
                  STORAGE_NAME=STORAGE_ACCOUNT, STORAGE_ACCOUNT=STORAGE_ACCOUNT,
                  USE_EXISTING=env("USE_EXISTING", "false"))
    for key, value in values.items():
        if "\n" in str(value) or "\r" in str(value):
            raise ValueError(f"{key} must be a single line")
        print(f"{key}={value}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stage", choices=["base", "models", "project", "application"], default="application")
    parser.add_argument("--shell", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shell:
        _shell()
    elif args.check:
        sys.exit(_check(args.stage))
    else:
        print(__doc__)
