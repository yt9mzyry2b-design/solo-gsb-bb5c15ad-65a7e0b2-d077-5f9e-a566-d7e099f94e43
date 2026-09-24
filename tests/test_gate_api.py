"""End-to-end tests for POST /gate/evaluate."""

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


DOC = {
    "subject": {
        "id": "u-123",
        "name": "Ada Lovelace",
        "age": 36,
        "isAdult": True,
        "nickname": None,
    },
    "emails": ["ada@example.com", "ada@analytical.engine"],
    "emptyObj": {},
    "score": 9.5,
}


def _commit(reveal, compact=False):
    req = {"document": DOC, "reveal": reveal}
    if compact:
        req["proofMode"] = "compact"
    return client.post("/commit", json=req).json()


def _gate(data, policy):
    body = copy.deepcopy(data)
    body["policy"] = policy
    return client.post("/gate/evaluate", json=body)


# ---------------------------------------------------------------------------
# Verification first — identical rules/errors as /verify
# ---------------------------------------------------------------------------


def test_gate_standard_and_compact_roundtrip():
    for compact in (False, True):
        data = _commit(["/subject/age", "/subject/name"], compact=compact)
        policy = {
            "all": [
                {"path": "/subject/name", "op": "eq", "value": "Ada Lovelace"},
                {"path": "/subject/age", "op": "gte", "value": 18},
            ]
        }
        resp = _gate(data, policy)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["result"] == "pass"
        assert body["root"] == data["root"]
        assert body["proofMode"] == ("compact" if compact else "standard")
        assert body["conditions"] == [
            {"path": "/subject/name", "op": "eq", "result": "pass"},
            {"path": "/subject/age", "op": "gte", "result": "pass"},
        ]


def test_gate_rejects_tampered_value_like_verify():
    data = _commit(["/subject/age"])
    body = copy.deepcopy(data)
    body["leaves"][0]["value"] = 37
    body["policy"] = {"path": "/subject/age", "op": "gte", "value": 18}
    resp = client.post("/gate/evaluate", json=body)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "ROOT_MISMATCH"
    assert err["loc"] == "leaves[0]"


def test_gate_rejects_bad_compact_proof_structure():
    data = _commit(["/subject/age", "/subject/name"], compact=True)
    body = copy.deepcopy(data)
    body["proofNodes"] = body["proofNodes"][:-1]
    body["policy"] = {"path": "/subject/age", "op": "gte", "value": 18}
    resp = client.post("/gate/evaluate", json=body)
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "BAD_PROOF"
    assert err["loc"] == "proofNodes"
    assert "missing" in err["detail"]


def test_gate_bad_hash_encoding_and_empty_leaves():
    resp = client.post(
        "/gate/evaluate",
        json={
            "root": "nope",
            "leaves": [],
            "policy": {"path": "/x", "op": "eq", "value": 1},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "BAD_HASH_ENCODING"
    assert resp.json()["error"]["loc"] == "root"

    data = _commit(["/subject/age"])
    body = copy.deepcopy(data)
    body["leaves"] = []
    body["policy"] = {"path": "/x", "op": "eq", "value": 1}
    resp = client.post("/gate/evaluate", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "EMPTY_REVEAL"


def test_proof_is_checked_before_policy():
    # A structurally bad proof fails even though the policy is also invalid.
    data = _commit(["/subject/age"])
    body = copy.deepcopy(data)
    body["leaves"][0]["value"] = "forged"
    body["policy"] = {"all": []}
    resp = client.post("/gate/evaluate", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "ROOT_MISMATCH"


# ---------------------------------------------------------------------------
# eq — canonical JSON byte comparison
# ---------------------------------------------------------------------------


def test_eq_canonical_bytes():
    data = _commit(["/subject/age"])
    ok = _gate(data, {"path": "/subject/age", "op": "eq", "value": 36})
    assert ok.json()["result"] == "pass"
    # 36 vs 36.0 differ canonically.
    diff = _gate(data, {"path": "/subject/age", "op": "eq", "value": 36.0})
    assert diff.json()["result"] == "fail"


def test_eq_string_null_and_empty_containers():
    data = _commit(["/subject/name", "/subject/nickname", "/emptyObj"])
    policy = {
        "all": [
            {"path": "/subject/name", "op": "eq", "value": "Ada Lovelace"},
            {"path": "/subject/nickname", "op": "eq", "value": None},
            {"path": "/emptyObj", "op": "eq", "value": {}},
            {"path": "/emptyObj", "op": "eq", "value": []},
        ]
    }
    resp = _gate(data, policy)
    assert [c["result"] for c in resp.json()["conditions"]] == [
        "pass", "pass", "pass", "fail"
    ]


def test_eq_bool_is_not_number_like():
    data = _commit(["/subject/isAdult"])
    assert _gate(data, {
        "path": "/subject/isAdult", "op": "eq", "value": True
    }).json()["result"] == "pass"
    assert _gate(data, {
        "path": "/subject/isAdult", "op": "eq", "value": 1
    }).json()["result"] == "fail"


# ---------------------------------------------------------------------------
# gte — finite non-boolean numbers only; non-numeric disclosure is false
# ---------------------------------------------------------------------------


def test_gte_numbers_including_float_threshold_and_value():
    data = _commit(["/subject/age", "/score"])
    policy = {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 36},
            {"path": "/subject/age", "op": "gte", "value": 36.0},
            {"path": "/subject/age", "op": "gte", "value": 37},
            {"path": "/score", "op": "gte", "value": 9.5},
        ]
    }
    results = [c["result"] for c in _gate(data, policy).json()["conditions"]]
    assert results == ["pass", "pass", "fail", "pass"]


def test_gte_non_numeric_disclosed_value_is_false():
    doc = {"b": True, "s": "36", "n": None, "e": []}
    data = client.post(
        "/commit", json={"document": doc, "reveal": ["/b", "/s", "/n", "/e"]}
    ).json()
    policy = {"all": [
        {"path": "/b", "op": "gte", "value": 0},
        {"path": "/s", "op": "gte", "value": 0},
        {"path": "/n", "op": "gte", "value": 0},
        {"path": "/e", "op": "gte", "value": 0},
    ]}
    resp = _gate(data, policy)
    assert resp.status_code == 200
    assert {c["path"]: c["result"] for c in resp.json()["conditions"]} == {
        "/b": "fail", "/s": "fail", "/n": "fail", "/e": "fail"
    }
    assert resp.json()["result"] == "fail"


# ---------------------------------------------------------------------------
# Unknown disclosure / three-valued logic
# ---------------------------------------------------------------------------


def test_undisclosed_path_is_unknown():
    data = _commit(["/subject/age"])
    resp = _gate(data, {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 18},
            {"path": "/subject/name", "op": "eq", "value": "Ada"},
        ]
    })
    body = resp.json()
    assert body["result"] == "unknown"
    assert body["conditions"][1]["result"] == "unknown"


def test_all_false_dominates_over_unknown():
    data = _commit(["/subject/age"])
    resp = _gate(data, {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 99},  # fail
            {"path": "/missing", "op": "eq", "value": 1},        # unknown
        ]
    })
    assert resp.json()["result"] == "fail"


def test_all_all_pass_and_all_unknown():
    data = _commit(["/subject/age"])
    assert _gate(data, {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 18},
            {"path": "/subject/age", "op": "eq", "value": 36},
        ]
    }).json()["result"] == "pass"
    assert _gate(data, {
        "all": [
            {"path": "/missing1", "op": "eq", "value": 1},
            {"any": [{"path": "/missing2", "op": "eq", "value": 2}]},
        ]
    }).json()["result"] == "unknown"


def test_any_true_dominates_over_unknown_and_false():
    data = _commit(["/subject/age"])
    resp = _gate(data, {
        "any": [
            {"path": "/subject/age", "op": "gte", "value": 99},  # fail
            {"path": "/subject/age", "op": "gte", "value": 18},  # pass
            {"path": "/missing", "op": "eq", "value": 1},        # unknown
        ]
    })
    assert resp.json()["result"] == "pass"


def test_any_all_fail_and_unknown_propagation():
    data = _commit(["/subject/age"])
    assert _gate(data, {
        "any": [
            {"path": "/subject/age", "op": "gte", "value": 99},
            {"path": "/subject/age", "op": "eq", "value": 99},
        ]
    }).json()["result"] == "fail"
    resp = _gate(data, {
        "any": [
            {"path": "/subject/age", "op": "gte", "value": 99},  # fail
            {"path": "/missing", "op": "eq", "value": 1},        # unknown
        ]
    })
    assert resp.json()["result"] == "unknown"


def test_nested_group_propagation_and_condition_order():
    data = _commit(["/subject/age", "/subject/name"])
    policy = {
        "any": [
            {"all": [
                {"path": "/subject/name", "op": "eq", "value": "Ada Lovelace"},
                {"path": "/subject/age", "op": "gte", "value": 65},
            ]},
            {"all": [
                {"path": "/missing", "op": "eq", "value": 1},
                {"path": "/subject/age", "op": "gte", "value": 18},
            ]},
        ]
    }
    body = _gate(data, policy).json()
    # first all: pass & fail -> fail; second all: unknown & pass -> unknown;
    # any(fail, unknown) -> unknown
    assert body["result"] == "unknown"
    assert [c["path"] for c in body["conditions"]] == [
        "/subject/name", "/subject/age", "/missing", "/subject/age"
    ]


def test_root_scalar_pointer_condition():
    data = client.post(
        "/commit", json={"document": 42, "reveal": [""]}
    ).json()
    resp = _gate(data, {"path": "", "op": "eq", "value": 42})
    assert resp.status_code == 200
    assert resp.json()["result"] == "pass"
    assert resp.json()["conditions"][0]["path"] == ""


# ---------------------------------------------------------------------------
# Policy structural validation — errors locate inside `policy`
# ---------------------------------------------------------------------------


def _invalid_policy(policy, loc=None, code="INVALID_REQUEST"):
    data = _commit(["/subject/age"])
    resp = _gate(data, policy)
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["code"] == code
    if loc is not None:
        assert err["loc"] == loc, err
    return err


def test_policy_must_be_present_and_be_a_node():
    data = _commit(["/subject/age"])
    resp = client.post(
        "/gate/evaluate",
        json={"root": data["root"], "leaves": data["leaves"]},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy"

    for bad in (5, "all", None, [{"all": []}]):
        assert _invalid_policy(bad, "policy")


def test_empty_groups_rejected_at_every_depth():
    _invalid_policy({"all": []}, "policy.all")
    _invalid_policy({"any": []}, "policy.any")
    _invalid_policy(
        {"all": [{"any": []}]}, "policy.all[0].any"
    )
    _invalid_policy(
        {"all": [{"all": [{"any": []}]}]}, "policy.all[0].all[0].any"
    )


def test_group_children_must_be_nonempty_node_array():
    _invalid_policy({"all": 5}, "policy.all")
    _invalid_policy({"any": {"path": "/a", "op": "eq", "value": 1}},
                    "policy.any")
    _invalid_policy({"all": [5]}, "policy.all[0]")
    _invalid_policy({"all": ["x"]}, "policy.all[0]")


def test_mixed_group_keys_and_mixed_node_fields():
    _invalid_policy({"all": [], "any": []}, "policy.all")
    _invalid_policy(
        {"all": [{"path": "/a", "op": "eq", "value": 1}], "path": "/a"},
        "policy.path",
    )
    _invalid_policy(
        {"any": [{"path": "/a", "op": "eq", "value": 1}], "op": "eq"},
        "policy.op",
    )


def test_unknown_fields_rejected():
    _invalid_policy({"all": [{"path": "/a", "op": "eq", "value": 1,
                             "extra": 2}]}, "policy.all[0].extra")
    _invalid_policy({"all": [{"path": "/a", "op": "eq", "value": 1}],
                     "extra": 1}, "policy.extra")
    _invalid_policy({"path": "/a", "op": "eq", "value": 1,
                     "extra": 2}, "policy.extra")


def test_condition_field_validation_and_locations():
    nested = lambda cond: {"all": [{"any": [cond]}]}
    base = "policy.all[0].any[0]"
    for field in ("path", "op", "value"):
        cond = {"path": "/a", "op": "eq", "value": 1}
        del cond[field]
        _invalid_policy(nested(cond), f"{base}.{field}")

    _invalid_policy(nested({"path": 7, "op": "eq", "value": 1}),
                    f"{base}.path")
    _invalid_policy(nested({"path": "/a", "op": "ne", "value": 1}),
                    f"{base}.op")
    _invalid_policy(nested({"path": "/a", "op": 9, "value": 1}),
                    f"{base}.op")
    _invalid_policy(
        nested({"path": "no-leading-slash", "op": "eq", "value": 1}),
        f"{base}.path",
        code="POINTER_SYNTAX",
    )
    _invalid_policy(
        nested({"path": "/a~", "op": "eq", "value": 1}),
        f"{base}.path",
        code="POINTER_SYNTAX",
    )


def test_gte_threshold_must_be_finite_nonboolean_number():
    for bad in (True, False, "18", None, [], {}, "abc"):
        _invalid_policy(
            {"path": "/subject/age", "op": "gte", "value": bad},
            "policy.value",
        )
    # Integers and floats are accepted (the request then evaluates).
    data = _commit(["/subject/age"])
    for threshold in (0, 18, 18.5, -3.0):
        resp = _gate(data, {"path": "/subject/age", "op": "gte",
                            "value": threshold})
        assert resp.status_code == 200, (threshold, resp.text)


def test_eq_threshold_accepts_any_json_value():
    data = _commit(["/subject/name", "/emptyObj"])
    policy = {"any": [
        {"path": "/subject/name", "op": "eq",
         "value": {"nested": [1, 2, {"k": "v"}]}},
        {"path": "/emptyObj", "op": "eq", "value": {}},
    ]}
    resp = _gate(data, policy)
    assert resp.status_code == 200
    assert resp.json()["result"] == "pass"


def test_gate_openapi_documents_route():
    spec = client.get("/openapi.json").json()
    assert "/gate/evaluate" in spec["paths"]
    operation = spec["paths"]["/gate/evaluate"]["post"]
    assert operation["operationId"] == "gateEvaluate"
    for name in (
        "StandardGateRequest",
        "CompactGateRequest",
        "GateEvaluateResponse",
        "GateConditionResult",
        "PolicyAllGroupModel",
        "PolicyAnyGroupModel",
        "GateConditionModel",
    ):
        assert name in spec["components"]["schemas"]


def test_index_lists_gate_endpoint():
    info = client.get("/").json()
    assert info["endpoints"]["gateEvaluate"] == "POST /gate/evaluate"
