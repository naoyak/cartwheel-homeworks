import pytest
from fastapi import HTTPException

from agent.auth import AuthContext


SHOPPER_1 = AuthContext(user_id=1, role="shopper")


def test_role_mismatch_rejected(world: dict):
    from server import app as server_app
    server_app._SESSIONS.clear()
    with pytest.raises(HTTPException) as exc:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))
    assert exc.value.status_code == 403


def test_token_for_another_session_rejected(world: dict):
    from server import app as server_app
    server_app._SESSIONS.clear()
    session_1 = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))
    session_2 = server_app.create_session(server_app.SessionCreate(user_id=9002, role="merchant"))
    with pytest.raises(HTTPException) as exc:
        _ = server_app._authorize(session_1["session_id"], f'Bearer {session_2["token"]}')
    assert exc.value.status_code == 403
