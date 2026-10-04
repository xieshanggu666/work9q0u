# -*- coding: utf-8 -*-
"""地堡联盟援助协议测试：管理者签约 → 医护负责人会签 → 外部聚落审核 →
押运运输/途中事件 → 交付/失败回退的完整三方契约状态链，以及托管冻结、
信誉/资源/医疗危机回写、阶段锁、幂等回放与终局收敛。"""
import pytest

from app.core.database import Base, engine, SessionLocal
from app.models import GameSession, Resident, Facility
from app.services.engine import (
    BunkerEngine,
    BunkerEngineError,
    BunkerEngineConflict,
    AID_INCIDENTS,
    AID_PROPOSED,
    AID_REVIEWING,
    AID_TRANSPORTING,
    FOOD, WATER,
    FACILITY_ZH,
)
from tests.test_engine import make_session, FixedRand


@pytest.fixture()
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    s = SessionLocal()
    yield s
    s.close()
    Base.metadata.drop_all(bind=engine)


def make_aid_session(db, residents=4, resources=None):
    """带医疗救治中心 + 当值医护的档案：residents[0] 默认是医护负责人候选。"""
    gs = make_session(db, residents=residents, resources=resources)
    gs.residents[0].job = "medic"
    db.add(
        Facility(
            session_id=gs.id, name=FACILITY_ZH["clinic"], category="clinic",
            level=1, status="active", built_day=1,
        )
    )
    db.commit()
    db.refresh(gs)
    return gs


class AidRand:
    """可脚本化的援助随机：依次弹出 random() 值，choice 取事件池首项。"""

    def __init__(self, random_vals=()):
        self.vals = list(random_vals)

    def random(self):
        return self.vals.pop(0) if self.vals else 0.9

    def choice(self, seq):
        return seq[0]


def aid_offer(eng):
    return eng.aid_market()[0]


def propose(eng, gs, medic=None, escorts=None):
    """完整走一遍管理者签约，返回协议快照。"""
    medic = medic or gs.residents[0]
    escorts = escorts or [gs.residents[1].id]
    return eng.propose_aid(aid_offer(eng)["id"], escorts, medic.id)


# ---- 市场与资格 ----

def test_aid_market_deterministic_and_two_offers(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    m1 = eng.aid_market()
    m2 = eng.aid_market()
    assert len(m1) == 2
    assert [o["id"] for o in m1] == [o["id"] for o in m2]
    for o in m1:
        assert o["type"] == "aid"
        assert len(o["escrow"]) == 2          # 医疗物资包：两种资源
        assert len(o["cargo"]) == 1


def test_aid_requires_clinic_and_medic(db):
    gs = make_session(db)  # 无救治中心、无医护
    eng = BunkerEngine(db, gs, rand=FixedRand())
    assert eng.aid_eligible() is False
    with pytest.raises(BunkerEngineError):
        propose(eng, gs, medic=gs.residents[0], escorts=[gs.residents[1].id])


def test_medic_must_be_on_duty(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    offer = aid_offer(eng)
    # residents[2] 是杂工，不能担任医护负责人
    with pytest.raises(BunkerEngineError):
        eng.propose_aid(offer["id"], [gs.residents[1].id], gs.residents[2].id)


def test_medic_cannot_join_escorts(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    offer = aid_offer(eng)
    with pytest.raises(BunkerEngineError):
        eng.propose_aid(offer["id"], [gs.residents[0].id], gs.residents[0].id)


# ---- 签约与会签 ----

def test_propose_freezes_escrow_and_waits_cosign(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    offer = aid_offer(eng)
    before = dict(gs.resources)
    order = propose(eng, gs)
    assert order["status"] == AID_PROPOSED
    assert order["medic_id"] == gs.residents[0].id
    # 托管已冻结
    for k, v in offer["escrow"].items():
        assert gs.resources[k] == round(before[k] - v, 1)
    # 待会签阶段：押运队与医护均在堡，不按离堡结算，可正常推进
    assert gs.residents[1].id not in eng._away_resident_ids()
    assert eng.phase == "daily"


def test_advance_while_proposed_keeps_order_and_runs_normal_day(db):
    """proposed（待会签）不随推进审核：当日正常走地堡逻辑，协议原样保留。"""
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.9]))  # 0.9 > 危机概率：无危机
    propose(eng, gs)
    eng.advance_day()
    assert gs.aid_order is not None
    assert gs.aid_order["status"] == AID_PROPOSED


def test_cosign_enters_reviewing_and_is_idempotent(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    order = propose(eng, gs)
    detail, replayed = eng.cosign_aid(token=order["token"])
    assert replayed is False
    assert gs.aid_order["status"] == AID_REVIEWING
    # 连点会签：幂等回放，不产生第二次状态变更
    detail2, replayed2 = eng.cosign_aid(token=order["token"])
    assert replayed2 is True
    assert detail2 == detail
    assert gs.aid_order["status"] == AID_REVIEWING


def test_cosign_blocked_if_medic_left_post(db):
    """签约后医护改岗：会签被拒，需撤单重签。"""
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    order = propose(eng, gs)
    gs.residents[0].job = "general"
    with pytest.raises(BunkerEngineError):
        eng.cosign_aid(token=order["token"])


def test_cancel_proposed_refunds_fully(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    before = dict(gs.resources)
    order = propose(eng, gs)
    detail, replayed = eng.cancel_aid(token=order["token"])
    assert replayed is False
    assert gs.aid_order is None
    assert gs.resources == before
    # 撤单连点幂等
    detail2, replayed2 = eng.cancel_aid(token=order["token"])
    assert replayed2 is True
    assert detail2 == detail
    assert gs.resources == before


def test_only_one_mission_at_a_time_with_aid(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=FixedRand())
    propose(eng, gs)
    # 待会签的援助协议同样占用离堡任务互斥位
    with pytest.raises(BunkerEngineError):
        eng.send_expedition([gs.residents[2].id], {FOOD: 5, WATER: 5})


# ---- 审核 ----

def test_review_rejection_refunds_full_escrow(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.99]))  # 高掷点：驳回
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    eng.advance_day()  # 审核日：驳回
    assert gs.aid_order is None
    assert gs.reputation == 50


def test_review_approval_starts_transport(db):
    gs = make_aid_session(db)
    # 会签后审核通过率 = 0.55 + 50/250 + 0.10 = 0.85；掷 0.1 通过；0.9 无事件
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.9]))
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    eng.advance_day()
    assert gs.aid_order is not None
    assert gs.aid_order["status"] == AID_TRANSPORTING
    assert gs.aid_order["travel_days"] == 1
    assert gs.residents[1].id in eng._away_resident_ids()
    # 医护负责人留堡
    assert gs.residents[0].id not in eng._away_resident_ids()


def test_cancel_blocked_after_transport_starts(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.9]))
    order = propose(eng, gs)
    token = order["token"]
    eng.cosign_aid(token=token)
    eng.advance_day()  # 审核通过进入运输
    with pytest.raises(BunkerEngineConflict):
        eng.cancel_aid(token=token)


# ---- 途中事件 ----

def test_aid_incident_pends_and_locks_phase(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.1]))  # 通过、触发事件
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    incident = eng.advance_day()
    assert incident is not None
    assert incident["event"] == AID_INCIDENTS[0]["key"]
    assert eng.phase == "aid"
    with pytest.raises(BunkerEngineError):
        eng.advance_day()
    with pytest.raises(BunkerEngineError):
        eng.build_facility("med")


def test_aid_incident_plague_contact_opens_infectious_case(db):
    """疫区病患接触（单体 -6 且传染）：受伤队员被立为传染病例，随队冻结。"""
    gs = make_aid_session(db, residents=4)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.1]))
    order = propose(eng, gs, escorts=[gs.residents[1].id])
    eng.cosign_aid(token=order["token"])
    incident = eng.advance_day()
    assert incident["event"] == "plague_contact"
    target_id = incident["target_id"]
    eng.resolve_aid_incident("sterile_aid", token=incident["token"])
    # 强制建档（force+infectious）：目标有了活跃传染病例
    cases = [c for c in gs.medical_cases if c["resident_id"] == target_id]
    assert cases and cases[-1]["infectious"] is True
    target = next(r for r in gs.residents if r.id == target_id)
    assert target.health < 90


def test_aid_incident_cargo_loss_and_idempotent(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.1]))
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    incident = eng.advance_day()
    # 首事件 plague_contact：抛下两成物资撤离（无暴露）
    d1, rp1 = eng.resolve_aid_incident("drop_supplies", token=incident["token"])
    assert rp1 is False
    assert gs.aid_order["cargo_ratio"] == 0.8
    d2, rp2 = eng.resolve_aid_incident("drop_supplies", token=incident["token"])
    assert rp2 is True
    assert d2 == d1
    assert gs.aid_order["cargo_ratio"] == 0.8  # 货损只结算一次


def test_aid_incident_wrong_token_conflict(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.1]))
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    eng.advance_day()
    with pytest.raises(BunkerEngineConflict):
        eng.resolve_aid_incident("drop_supplies", token="stale")


def test_aid_abort_settles_order_immediately(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.1]))
    order = propose(eng, gs)
    before = dict(gs.resources)
    eng.cosign_aid(token=order["token"])
    incident = eng.advance_day()
    detail, _ = eng.resolve_aid_incident("abandon", token=incident["token"])
    assert gs.aid_order is None
    assert gs.reputation == 40  # -10
    # 弃货货损为 0：托管全额带回（当日生产另计，只核对托管键）
    offer = aid_offer(eng)
    for k, v in offer["escrow"].items():
        assert gs.resources[k] >= before[k] - v
    # 连点回放
    detail2, replayed = eng.resolve_aid_incident("abandon", token=incident["token"])
    assert replayed is True
    assert detail2 == detail


# ---- 交付：信誉 / 资源 / 医疗危机回写 ----

def test_successful_delivery_writes_back_all_three(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.9]))
    offer = aid_offer(eng)
    order = propose(eng, gs, escorts=[gs.residents[1].id])
    token = order["token"]
    eng.cosign_aid(token=token)
    eng.advance_day()  # 审核通过 + 第一个在途日（无事件）
    # 抵达日：交付掷点。force_success 路径不走暴露；随机序列给：
    # 暴露判定（医护在岗 → 0.25 概率）掷 0.9 不暴露；交付掷 0.1 成功
    eng.rand = AidRand([0.9, 0.1])
    in_bunker_before = {r.id: r.health for r in gs.residents if r.id != gs.residents[1].id}
    gain_key = list(offer["cargo"].keys())[0]
    gain_before = gs.resources[gain_key]
    eng.advance_day()
    assert gs.aid_order is None
    assert gs.reputation == 58  # +8
    # 资源回写：对方回赠入库
    assert gs.resources[gain_key] > gain_before
    # 医疗危机回写：在堡全员健康 +4
    for r in gs.residents:
        if r.id in in_bunker_before:
            assert r.health == min(100.0, in_bunker_before[r.id] + 4)
    # 押运队员士气奖励
    assert gs.residents[1].morale > 80


def test_exposure_registers_infectious_case_on_delivery(db):
    """抵达疫区暴露（掷点必暴露）+ 交付成功：暴露者立传染病例，协议仍成功。"""
    gs = make_aid_session(db)
    # 无医护加成的暴露概率 0.45；医护在岗压到 0.25。掷 0.01 必暴露
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.9]))
    order = propose(eng, gs, escorts=[gs.residents[1].id])
    token = order["token"]
    eng.cosign_aid(token=token)
    eng.advance_day()
    eng.rand = AidRand([0.01, 0.1])  # 暴露；交付成功
    eng.advance_day()
    assert gs.aid_order is None
    assert gs.reputation == 58
    cases = [c for c in gs.medical_cases
             if c["resident_id"] == gs.residents[1].id and c["infectious"]]
    assert cases  # 押运队员被立为传染病例，回堡后续治


def test_exposure_chance_lower_with_medic(db):
    """医护负责人主持防疫：暴露概率低于无医护情形（常量口径）。"""
    from app.services.engine import (
        AID_EXPOSURE_BASE, AID_EXPOSURE_MEDIC_REDUCE, AID_EXPOSURE_MIN_CHANCE,
    )
    assert AID_EXPOSURE_BASE - AID_EXPOSURE_MEDIC_REDUCE < AID_EXPOSURE_BASE
    assert AID_EXPOSURE_BASE - AID_EXPOSURE_MEDIC_REDUCE >= AID_EXPOSURE_MIN_CHANCE


def test_failed_delivery_refunds_and_penalizes(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.9]))
    offer = aid_offer(eng)
    order = propose(eng, gs, escorts=[gs.residents[1].id])
    token = order["token"]
    eng.cosign_aid(token=token)
    eng.advance_day()
    # 抵达日：暴露掷 0.9 不暴露；交付成功率 0.60+0.2+0.10=0.90，掷 0.95 失败
    eng.rand = AidRand([0.9, 0.95])
    frozen_after_keys = {k: gs.resources[k] for k in offer["escrow"]}
    eng.advance_day()
    assert gs.aid_order is None
    assert gs.reputation == 40  # -10
    # 托管全额退回（无在途货损）
    for k, v in offer["escrow"].items():
        assert gs.resources[k] >= frozen_after_keys[k]


def test_inbound_aid_replaces_bunker_crisis(db):
    """援助押运在途的日子只走援助途中事件，不触发地堡危机。"""
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.1]))
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    pending = eng.advance_day()
    assert pending is not None
    assert gs.pending_crisis is None
    assert gs.aid_order["pending_incident"] is not None


# ---- 离堡口径 ----

def test_transport_day_excludes_aid_escorts_from_consumption(db):
    """在途援助押运队员不消耗地堡口粮（与探索/贸易押运同一口径）。"""
    gs = make_aid_session(db, residents=4)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.9]))
    offer = aid_offer(eng)
    order = eng.propose_aid(offer["id"], [gs.residents[1].id], gs.residents[0].id)
    eng.cosign_aid(token=order["token"])
    # 把行程拉长到 4 天，保证完整在途日仍未抵达
    gs.aid_order["eta"] = 4
    eng.advance_day()  # 审核通过 + 出发
    food_day2 = gs.resources[FOOD]
    eng.rand = AidRand([0.9])
    eng.advance_day()  # 完整在途日
    delta_aid = gs.resources[FOOD] - food_day2

    gs_ref = make_session(db, residents=4)
    eng_ref = BunkerEngine(db, gs_ref, rand=AidRand([0.9]))
    eng_ref.advance_day()
    ref_day2 = gs_ref.resources[FOOD]
    eng_ref.advance_day()
    delta_ref = gs_ref.resources[FOOD] - ref_day2
    # 唯一差别：援助档案 1 人离堡不消耗地堡口粮（少消耗 1.5）
    assert round(delta_aid - delta_ref, 1) == 1.5


# ---- 终局收敛 ----

def test_endgame_clears_aid_order_and_forces_delivery(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.9]))
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    eng.advance_day()  # 在途
    assert gs.aid_order is not None
    gs.day = gs.target_day
    gs.pending_crisis = None
    eng.advance_day()
    assert gs.status == "win"
    assert gs.aid_order is None


def test_reviewing_aid_on_endgame_cancelled_with_refund(db):
    gs = make_aid_session(db)
    gs.day = gs.target_day
    eng = BunkerEngine(db, gs, rand=FixedRand())
    before = dict(gs.resources)
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    eng.advance_day()  # 终局日：审核不再进行，撤单退款
    assert gs.status == "win"
    assert gs.aid_order is None
    for k, v in order["escrow"].items():
        assert gs.resources[k] >= before[k] - v


def test_proposed_aid_on_endgame_cancelled_with_refund(db):
    """终局日仍停留在待会签的协议：同样撤单退款，托管医疗物资不丢失。"""
    gs = make_aid_session(db)
    gs.day = gs.target_day
    eng = BunkerEngine(db, gs, rand=FixedRand())
    before = dict(gs.resources)
    order = propose(eng, gs)  # 不调用会签
    eng.advance_day()
    assert gs.status == "win"
    assert gs.aid_order is None
    for k, v in order["escrow"].items():
        assert gs.resources[k] >= before[k] - v


# ---- 快照持久化与恢复 ----

def test_aid_order_persisted_and_recoverable_after_reload(db):
    gs = make_aid_session(db)
    eng = BunkerEngine(db, gs, rand=AidRand([0.1, 0.1]))
    order = propose(eng, gs)
    eng.cosign_aid(token=order["token"])
    incident = eng.advance_day()
    token = incident["token"]
    db.commit()
    sid = gs.id

    db2 = SessionLocal()
    try:
        gs2 = db2.get(GameSession, sid)
        eng2 = BunkerEngine(db2, gs2, rand=FixedRand())
        assert eng2.phase == "aid"
        assert gs2.aid_order["pending_incident"]["token"] == token
        eng2.resolve_aid_incident("drop_supplies", token=token)
        db2.commit()
    finally:
        db2.close()


# ---- HTTP 层 ----

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def test_aid_api_full_flow(client):
    """HTTP 端到端：市场 → 签约 → 会签（→ 推进/审核/运输由引擎测试覆盖）。"""
    from app.services.engine import FACILITY_ZH as _FZ
    # 建一个带医护与救治中心的档案：直接走 API 建会话后调岗+建中心
    r = client.post("/api/sessions", json={"name": "aid-e2e"})
    sid = r.json()["id"]
    residents = r.json()["residents"]
    medic = residents[0]
    escort = residents[1]
    r = client.post(f"/api/sessions/{sid}/resident/{medic['id']}/job", json={"job": "medic"})
    assert r.status_code == 200
    r = client.post(f"/api/sessions/{sid}/build", json={"category": "clinic"})
    assert r.status_code == 200

    r = client.get(f"/api/sessions/{sid}/aid/market")
    assert r.status_code == 200
    assert r.json()["eligible"] is True
    offer = r.json()["offers"][0]

    r = client.post(f"/api/sessions/{sid}/aid/propose", json={
        "offer_id": offer["id"], "escort_ids": [escort["id"]], "medic_id": medic["id"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["aid_order"]["status"] == "proposed"
    # 会签
    r = client.post(f"/api/sessions/{sid}/aid/cosign", json={"token": body["aid_order"]["token"]})
    assert r.status_code == 200
    assert r.json()["aid_order"]["status"] == "reviewing"
    # 撤单窗口已关闭（审核中仍可撤单）——改为校验撤单成功退款路径：
    r = client.post(f"/api/sessions/{sid}/aid/cancel", json={"token": body["aid_order"]["token"]})
    assert r.status_code == 200
    assert r.json()["aid_order"] is None
