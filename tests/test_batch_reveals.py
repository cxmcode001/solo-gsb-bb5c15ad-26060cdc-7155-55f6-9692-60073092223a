"""POST /events/{event_id}/reveals/batch 原子批量揭示测试。

SQLite 文件库 + 模拟时钟（与 test_api.py 相同手法），覆盖：
  - 揭示期内按输入顺序批量写入并返回 count
  - 盐格式非法 / 请求内重复 / 无匹配承诺 / 已揭示 的逐项 409 冲突清单
  - 混合冲突一次报全，且整体回滚不留部分揭示
  - 数量边界（0、101）422
  - 揭示期未开始 / 已结束、事件不存在 404 仍生效
  - 并发竞争下仅首次揭示生效（唯一约束兜底，整批回滚）
  - 与单个揭示端点共享存储，揭示进入时间线
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from uuid import UUID

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_batch_reveal_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/batch_reveal.db"

import pytest  # noqa: E402
import sqlalchemy.orm as orm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_module  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.hashing import compute_commitment  # noqa: E402
from app.models import Commitment, Reveal  # noqa: E402

T0 = datetime(2030, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
COMMIT_DL = T0 + timedelta(hours=1)
REVEAL_DL = T0 + timedelta(hours=2)


class MutableClock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t


clock = MutableClock(T0)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    clock.t = T0
    monkeypatch.setattr(main_module, "now_utc", clock)
    yield


client = TestClient(main_module.app)


def text_of(i: int) -> str:
    return f"prediction-{i}"


def salt_of(i: int) -> str:
    return f"salt-{i}"


def h(i: int) -> str:
    return compute_commitment(text_of(i), salt_of(i))


def item(i: int) -> dict:
    return {"original_text": text_of(i), "salt": salt_of(i)}


def create_event_with_commitments(indices) -> str:
    """创建事件（截止时间相对 T0），在 commit 阶段批量提交给定序号的承诺。"""
    with client:
        r = client.post("/events", json={
            "title": "批量揭示事件",
            "commit_deadline": COMMIT_DL.isoformat(),
            "reveal_deadline": REVEAL_DL.isoformat(),
        })
        assert r.status_code == 201, r.text
        eid = r.json()["id"]
        r = client.post(
            f"/events/{eid}/commitments/batch",
            json={"commitment_hashes": [h(i) for i in indices]},
        )
        assert r.status_code == 201, r.text
    return eid


def reveal_count(event_id: str) -> int:
    return client.get(f"/events/{event_id}").json()["reveal_count"]


def batch_reveal(event_id: str, reveals):
    return client.post(
        f"/events/{event_id}/reveals/batch",
        json={"reveals": reveals},
    )


@pytest.fixture
def event_id() -> str:
    eid = create_event_with_commitments(range(1, 6))
    clock.t = COMMIT_DL  # 进入揭示期
    return eid


def test_batch_reveals_in_input_order(event_id):
    r = batch_reveal(event_id, [item(3), item(1), item(2)])  # 故意打乱承诺顺序
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["count"] == 3
    out = body["reveals"]
    # 严格按输入顺序返回，与重算哈希/承诺一一对应
    assert [x["recomputed_hash"] for x in out] == [h(3), h(1), h(2)]
    assert [x["commitment_hash"] for x in out] == [h(3), h(1), h(2)]
    assert len({x["id"] for x in out}) == 3
    assert all(x["event_id"] == event_id for x in out)
    assert all(x["verified"] is True for x in out)
    assert all(x["hash_scheme"] == main_module.HASH_SCHEME for x in out)
    assert reveal_count(event_id) == 3

    # 承诺列表反映揭示状态
    r = client.get(f"/events/{event_id}/commitments")
    revealed = {c["commitment_hash"]: c["revealed"] for c in r.json()}
    assert revealed[h(1)] is True
    assert revealed[h(3)] is True
    assert revealed[h(4)] is False


def test_batch_reveals_appear_in_timeline(event_id):
    r = batch_reveal(event_id, [item(1), item(2)])
    assert r.status_code == 201, r.text
    tl = client.get(f"/events/{event_id}/timeline").json()
    types = [e["type"] for e in tl["entries"]]
    assert types == [
        "event_created",
        "commitment_submitted",
        "commitment_submitted",
        "commitment_submitted",
        "commitment_submitted",
        "commitment_submitted",
        "reveal_accepted",
        "reveal_accepted",
    ]


def test_batch_reveal_single_item_ok(event_id):
    r = batch_reveal(event_id, [item(5)])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["reveals"][0]["recomputed_hash"] == h(5)


def test_batch_reveal_invalid_format(event_id):
    r = batch_reveal(event_id, [
        item(1),
        {"original_text": "空盐", "salt": ""},
        {"original_text": "过长盐", "salt": "x" * 257},
        item(2),
    ])
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "BATCH_REVEAL_CONFLICT"
    conflicts = detail["conflicts"]
    assert [(c["index"], c["type"]) for c in conflicts] == [
        (1, "INVALID_FORMAT"),
        (2, "INVALID_FORMAT"),
    ]
    # 合法项也随整批回滚，零写入
    assert reveal_count(event_id) == 0


def test_batch_reveal_duplicate_within_request(event_id):
    r = batch_reveal(event_id, [item(1), item(2), item(1), item(2), item(1)])
    assert r.status_code == 409, r.text
    conflicts = r.json()["detail"]["conflicts"]
    assert [(c["index"], c["type"]) for c in conflicts] == [
        (2, "DUPLICATE_IN_REQUEST"),
        (3, "DUPLICATE_IN_REQUEST"),
        (4, "DUPLICATE_IN_REQUEST"),
    ]
    assert conflicts[0]["recomputed_hash"] == h(1)
    assert reveal_count(event_id) == 0


def test_batch_reveal_no_matching_commitment(event_id):
    # item(99) 的哈希在本事件中无对应承诺
    r = batch_reveal(event_id, [item(1), item(99), item(2)])
    assert r.status_code == 409
    conflicts = r.json()["detail"]["conflicts"]
    assert conflicts == [{"index": 1, "type": "NO_MATCHING_COMMITMENT", "recomputed_hash": h(99)}]
    assert reveal_count(event_id) == 0


def test_batch_reveal_wrong_text_and_salt(event_id):
    # 原文正确但盐错误 -> 重算哈希不同 -> 无匹配
    r = batch_reveal(event_id, [{
        "original_text": text_of(1),
        "salt": "wrong-salt",
    }])
    assert r.status_code == 409
    assert r.json()["detail"]["conflicts"][0]["type"] == "NO_MATCHING_COMMITMENT"
    assert reveal_count(event_id) == 0


def test_batch_reveal_already_revealed(event_id):
    # 单个端点先揭示 item(1)
    r = client.post(f"/events/{event_id}/reveals", json=item(1))
    assert r.status_code == 201, r.text

    r = batch_reveal(event_id, [item(3), item(1), item(2)])
    assert r.status_code == 409
    conflicts = r.json()["detail"]["conflicts"]
    assert conflicts == [{"index": 1, "type": "DUPLICATE_REVEAL", "recomputed_hash": h(1)}]
    # 未冲突的 item(3)/item(2) 不得写入
    assert reveal_count(event_id) == 1


def test_batch_reveal_first_occurrence_gets_db_conflict(event_id):
    # item(1) 已被揭示且在请求中重复三次：首个下标记已揭示，其余记请求内重复
    assert client.post(f"/events/{event_id}/reveals", json=item(1)).status_code == 201
    r = batch_reveal(event_id, [item(1), item(1), item(1)])
    assert r.status_code == 409
    assert [(c["index"], c["type"]) for c in r.json()["detail"]["conflicts"]] == [
        (0, "DUPLICATE_REVEAL"),
        (1, "DUPLICATE_IN_REQUEST"),
        (2, "DUPLICATE_IN_REQUEST"),
    ]


def test_batch_reveal_mixed_conflicts_all_reported_and_rolled_back(event_id):
    # item(2) 已被揭示
    assert client.post(f"/events/{event_id}/reveals", json=item(2)).status_code == 201

    # 下标: 0 item1(合法) 1 空盐(格式非法) 2 item3(合法)
    #       3 item1(请求内重复) 4 item2(已揭示) 5 item99(无匹配)
    r = batch_reveal(event_id, [
        item(1),
        {"original_text": "空盐", "salt": ""},
        item(3),
        item(1),
        item(2),
        item(99),
    ])
    assert r.status_code == 409, r.text
    conflicts = r.json()["detail"]["conflicts"]
    assert [(c["index"], c["type"]) for c in conflicts] == [
        (1, "INVALID_FORMAT"),
        (3, "DUPLICATE_IN_REQUEST"),
        (4, "DUPLICATE_REVEAL"),
        (5, "NO_MATCHING_COMMITMENT"),
    ]
    # 合法项 item1/item3 也随整批回滚；仍只有先前 item2 一条揭示
    assert reveal_count(event_id) == 1

    # 修正全部冲突后可以整批成功
    r = batch_reveal(event_id, [item(1), item(3)])
    assert r.status_code == 201, r.text
    assert [x["recomputed_hash"] for x in r.json()["reveals"]] == [h(1), h(3)]


def test_batch_reveal_size_limits():
    # 新事件 + 100 个承诺，验证恰好 100 组允许
    eid = create_event_with_commitments(range(1000, 1100))
    clock.t = COMMIT_DL

    r = batch_reveal(eid, [])
    assert r.status_code == 422

    r = batch_reveal(eid, [item(1000 + i) for i in range(101)])
    assert r.status_code == 422

    r = batch_reveal(eid, [item(1000 + i) for i in range(100)])
    assert r.status_code == 201, r.text
    assert r.json()["count"] == 100
    assert reveal_count(eid) == 100


def test_batch_reveal_wrong_body_shape_returns_422(event_id):
    r = client.post(f"/events/{event_id}/reveals/batch", json={"reveals": "not-a-list"})
    assert r.status_code == 422
    r = client.post(f"/events/{event_id}/reveals/batch", json={})
    assert r.status_code == 422
    r = client.post(
        f"/events/{event_id}/reveals/batch",
        json={"reveals": [{"original_text": "缺盐"}]},
    )
    assert r.status_code == 422
    assert reveal_count(event_id) == 0


def test_batch_reveal_before_reveal_phase(event_id):
    clock.t = T0  # 回到 commit 阶段
    r = batch_reveal(event_id, [item(1)])
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "REVEAL_PHASE_NOT_STARTED"
    assert reveal_count(event_id) == 0


def test_batch_reveal_after_reveal_deadline(event_id):
    clock.t = REVEAL_DL
    r = batch_reveal(event_id, [item(1)])
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "REVEAL_PHASE_CLOSED"
    assert reveal_count(event_id) == 0


def test_batch_reveal_unknown_event_returns_404():
    clock.t = COMMIT_DL
    missing = "00000000-0000-0000-0000-000000000000"
    r = batch_reveal(missing, [item(1)])
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "EVENT_NOT_FOUND"


def test_batch_and_single_reveal_endpoint_share_storage(event_id):
    assert batch_reveal(event_id, [item(1), item(2)]).status_code == 201

    # 单个端点再揭示批内已揭示的承诺 -> 原有的单项重复语义
    r = client.post(f"/events/{event_id}/reveals", json=item(1))
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "DUPLICATE_REVEAL"

    # 单个端点揭示 item(3) 后，批量端点也应看到它已被揭示
    assert client.post(f"/events/{event_id}/reveals", json=item(3)).status_code == 201
    r = batch_reveal(event_id, [item(3), item(4)])
    assert r.status_code == 409
    conflicts = r.json()["detail"]["conflicts"]
    assert [(c["index"], c["type"]) for c in conflicts] == [(0, "DUPLICATE_REVEAL")]
    assert reveal_count(event_id) == 3


def test_batch_reveal_concurrent_first_reveal_wins(event_id, monkeypatch):
    """模拟并发竞争：预检通过后、提交前另一事务抢先揭示 item(1)。

    唯一约束导致本批提交失败，整批回滚并给出逐项清单，仅抢先事务的揭示生效。
    """
    original_commit = orm.Session.commit
    state = {"raced": False}

    def racing_commit(self):
        if not state["raced"]:
            state["raced"] = True
            # 另一事务抢先揭示 item(1)；直接调用未补丁的 commit 避免递归
            other = SessionLocal()
            try:
                commitment = (
                    other.query(Commitment)
                    .filter(
                        Commitment.event_id == UUID(event_id),
                        Commitment.commitment_hash == h(1),
                    )
                    .one()
                )
                other.add(Reveal(
                    event_id=commitment.event_id,
                    commitment_id=commitment.id,
                    original_text=text_of(1),
                    salt=salt_of(1),
                    recomputed_hash=h(1),
                ))
                original_commit(other)
            finally:
                other.close()
        original_commit(self)

    monkeypatch.setattr(orm.Session, "commit", racing_commit)

    r = batch_reveal(event_id, [item(1), item(2)])
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "BATCH_REVEAL_CONFLICT"
    conflicts = detail["conflicts"]
    assert [(c["index"], c["type"]) for c in conflicts] == [(0, "DUPLICATE_REVEAL")]
    # 整批回滚：item(2) 未写入；仅并发对手的 item(1) 揭示存在
    assert reveal_count(event_id) == 1
    r = client.get(f"/events/{event_id}/commitments")
    revealed = {c["commitment_hash"]: c["revealed"] for c in r.json()}
    assert revealed[h(1)] is True
    assert revealed[h(2)] is False
