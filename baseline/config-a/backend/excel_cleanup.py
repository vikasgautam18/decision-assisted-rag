"""Remove orphaned per-upload Excel agents (left behind if the backend restarted before the session TTL).

Usage: python3 -m backend.excel_cleanup          # list
       python3 -m backend.excel_cleanup --delete  # delete (run only while no uploads are active)
"""
import argparse
import re

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential

from common import config as C

SESSION_AGENT = re.compile(r"^excel-[0-9a-f]{12}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    project = AIProjectClient(endpoint=C.project_endpoint(), credential=AzureCliCredential())
    names = [x.name for x in project.agents.list() if SESSION_AGENT.match(x.name)]
    for n in names:
        if a.delete:
            project.agents.delete(n, force=True)
        print(("deleted " if a.delete else "orphan ") + n)
    print(f"{len(names)} session agent(s)")


if __name__ == "__main__":
    main()
