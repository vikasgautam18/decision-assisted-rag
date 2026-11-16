import pytest
from fastapi.testclient import TestClient

from app.live import create_live_app
from app.sessions import ConversationBusy, ConversationNotFound, SessionStore


class FakeRunner:
    def __init__(self):
        self.histories = []

    async def run(self, question, history):
        self.histories.append(history)
        return {
            "status": "completed",
            "behavior": "abstain",
            "answer": "No matching evidence in this fixture.",
            "citations": [],
            "figures": [],
            "semantic_verification": "not_run",
            "execution_mode": "test",
            "trace_id": "test",
        }


def test_cookie_conversation_ownership_and_only_user_intent_passed_to_pipeline(tmp_path):
    runner = FakeRunner()
    application = create_live_app(tmp_path, runner=runner)
    with (
        TestClient(application, headers={"origin": "http://testserver"}) as first,
        TestClient(application, headers={"origin": "http://testserver"}) as other,
    ):
        response = first.post("/api/chat", json={"question": "NOVA-3 range?"})
        assert response.status_code == 200
        cid = response.json()["conversation_id"]
        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=strict" in cookie
        assert other.get(f"/api/conversations/{cid}").status_code == 404
        assert (
            other.post(
                "/api/chat", json={"question": "followup", "conversation_id": cid}
            ).status_code
            == 404
        )
        assert (
            first.post(
                "/api/chat", json={"question": "what about petrol?", "conversation_id": cid}
            ).status_code
            == 200
        )
        assert runner.histories == [[], ["NOVA-3 range?"]]


def test_live_api_rejects_foreign_kb_and_resource_controls(tmp_path):
    with TestClient(
        create_live_app(tmp_path, runner=FakeRunner()), headers={"origin": "http://testserver"}
    ) as client:
        assert (
            client.post("/api/chat", json={"question": "Q", "kb": "dossier-kb"}).status_code == 403
        )
        assert (
            client.post("/api/chat", json={"question": "Q", "index": "dossier-chunks"}).status_code
            == 422
        )
        assert client.get("/api/figures/unknown").status_code == 404
        assert client.get("/health").json()["phase_3_authorized"] is False


def test_homepage_has_no_notice_banner(tmp_path):
    with TestClient(create_live_app(tmp_path, runner=FakeRunner())) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'aria-label="Application status"' not in response.text
    assert 'class="notice"' not in response.text
    assert "Phase 1: six-dossier" not in response.text


def test_conversation_lock_and_failed_turn_release(tmp_path):
    store = SessionStore(tmp_path / "conversations.db")
    sid, _ = store.session("")
    cid, history = store.begin(sid, None)
    assert history == []
    with pytest.raises(ConversationBusy):
        store.begin(sid, cid)
    store.finish(sid, cid, "failed", None)
    assert store.history(sid, cid) == []
    store.begin(sid, cid)


def test_cookie_tampering_and_restart_do_not_reuse_owners(tmp_path):
    store = SessionStore(tmp_path / "conversations.db")
    sid, cookie = store.session("")
    assert store.session(cookie)[0] == sid
    assert store.session(cookie[:-1] + ("0" if cookie[-1] != "0" else "1"))[0] != sid
    cid, _ = store.begin(sid, None)
    store.finish(sid, cid, "Q", None)
    restarted = SessionStore(tmp_path / "conversations.db")
    new_sid, _ = restarted.session(cookie)
    with pytest.raises(ConversationNotFound):
        restarted.history(new_sid, cid)


def test_error_not_written_as_a_semantic_assistant_turn(tmp_path):
    store = SessionStore(tmp_path / "conversations.db")
    sid, _ = store.session("")
    cid, _ = store.begin(sid, None)
    store.finish(sid, cid, "Q", {"status": "error"})
    assert store.history(sid, cid) == []
