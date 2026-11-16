"""The published repository must not carry local Azure values, addresses or personal data."""

import ipaddress
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
RESOURCE_MANIFEST = ROOT / "manifests/azure-resources.json"
PLACEHOLDER_VALUES = {
    "subscription_id": "00000000-0000-0000-0000-000000000000",
    "resource_group": "example-resource-group",
    "storage_account": "examplestorageacct",
}
ENDPOINT_FIELDS = ("search_endpoint", "di_endpoint", "ai_endpoint")
PLACEHOLDER_MANIFEST = {
    **PLACEHOLDER_VALUES,
    "search_endpoint": "https://example-search.search.windows.net",
    "di_endpoint": "https://example-di.cognitiveservices.azure.com",
    "ai_endpoint": "https://example-ai.cognitiveservices.azure.com",
}

IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
EMAIL_DOMAIN = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
HOME_PATH = re.compile(r"/home/[\w.-]+|/Users/[\w.-]+")
DOCUMENTATION_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)
SAFE_EMAIL_DOMAINS = ("example.com", "example.org", "users.noreply.github.com")
SAFE_EMAIL_SUFFIXES = (".invalid", ".test")


def non_placeholder_fields(manifest: dict) -> list[str]:
    fields = [name for name, value in PLACEHOLDER_VALUES.items() if manifest.get(name) != value]
    for name in ENDPOINT_FIELDS:
        host = urlparse(manifest.get(name) or "").hostname or ""
        if not host.startswith("example-"):
            fields.append(name)
    return fields


def find_leaks(text: str) -> list[str]:
    leaks = []
    for candidate in IPV4.findall(text):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        documentation = any(address in network for network in DOCUMENTATION_NETWORKS)
        if not (address.is_loopback or address.is_unspecified or documentation):
            leaks.append("IPv4 address")
    for domain in EMAIL_DOMAIN.findall(text):
        domain = domain.lower()
        if domain not in SAFE_EMAIL_DOMAINS and not domain.endswith(SAFE_EMAIL_SUFFIXES):
            leaks.append("email address")
    if HOME_PATH.search(text):
        leaks.append("home directory path")
    return leaks


def git_output(*args: str) -> bytes | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run([git, *args], cwd=ROOT, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def is_skip_worktree(path: Path) -> bool:
    flags = git_output("ls-files", "-v", "--", str(path.relative_to(ROOT)))
    return flags is not None and flags[:1] in (b"S", b"s")


def publishable_files() -> list[Path]:
    listing = git_output("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if listing is None:
        pytest.skip("git metadata is required to list the files that would be published")
    paths = (ROOT / name for name in listing.decode().split("\0") if name)
    return sorted(path for path in paths if path.is_file() and path != THIS_FILE)


def test_placeholder_manifest_passes_the_check():
    assert non_placeholder_fields(PLACEHOLDER_MANIFEST) == []


@pytest.mark.parametrize("field", [*PLACEHOLDER_VALUES, *ENDPOINT_FIELDS])
def test_each_real_looking_manifest_value_is_flagged(field):
    real_looking = "https://contoso.search.windows.net" if field in ENDPOINT_FIELDS else "contoso01"
    assert non_placeholder_fields({**PLACEHOLDER_MANIFEST, field: real_looking}) == [field]


def test_missing_manifest_field_is_flagged():
    incomplete = {k: v for k, v in PLACEHOLDER_MANIFEST.items() if k != "search_endpoint"}
    assert non_placeholder_fields(incomplete) == ["search_endpoint"]


def test_committed_resource_manifest_holds_only_placeholders():
    if is_skip_worktree(RESOURCE_MANIFEST):
        pytest.skip("local Azure values are deliberately kept out of git with skip-worktree")
    assert non_placeholder_fields(json.loads(RESOURCE_MANIFEST.read_text())) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("connect to 10.20.30.40 now", ["IPv4 address"]),
        ("allow 203.0.113.9, 127.0.0.1 and 0.0.0.0", []),
        ("not an address: 256.1.1.1", []),
        ("version 1.2.3, port 8765", []),
        ("mail someone@gmail.com", ["email address"]),
        ("mail a@example.invalid, b@users.noreply.github.com", []),
        ("see /home/someone/project", ["home directory path"]),
        ("see /Users/someone/project", ["home directory path"]),
    ],
)
def test_leak_detector(text, expected):
    assert find_leaks(text) == expected


def test_publishable_files_contain_no_addresses_emails_or_home_paths():
    leaks = {}
    for path in publishable_files():
        raw = path.read_bytes()
        if b"\0" in raw[:8192]:
            continue
        for number, line in enumerate(raw.decode("utf-8", errors="ignore").splitlines(), 1):
            if found := find_leaks(line):
                leaks[f"{path.relative_to(ROOT)}:{number}"] = sorted(set(found))
    assert leaks == {}
