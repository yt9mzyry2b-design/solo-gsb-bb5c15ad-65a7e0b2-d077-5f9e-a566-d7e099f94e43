"""End-to-end tests for POST /gate/evaluate."""

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


DOC = {
    "issuer": "https://id.example",
    "subject": {
        "id": "u-123",
        "name": "Ada Lovelace",
        "age": 36,
        "isAdult": True,
        "nickname": None,
    },
    "emails": ["ada@example.com", "ada@analytical.engine"],
    "emptyObj": {},
    "emptyArr": [],
}

REVEAL = ["/subject/age", "/subject/name", "/subject/isAdult", "/emptyObj"]


def _commit(reveal=REVEAL, proof_mode=None, document=DOC):
    payload = {"document": document, "reveal": reveal}
    if proof_mode is not None:
        payload["proofMode"] = proof_mode
    resp = client.post("/commit", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _gate(committed, policy, **overrides):
    payload = {
        key: copy.deepcopy(value)
        for key, value in committed.items()
        if key != "proofMode"
    }
    payload["policy"] = policy
    payload.update(overrides)
    return client.post("/gate/evaluate", json=payload)


def _condition_results(body):
    return {entry["loc"]: entry["result"] for entry in body["conditions"]}


# ---------------------------------------------------------------------------
# Happy paths: standard and compact proofs, nested groups
# ---------------------------------------------------------------------------


def test_gate_pass_with_nested_all_any_standard_proof():
    committed = _commit()
    policy = {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 18},
            {
                "any": [
                    {"path": "/subject/name", "op": "eq", "value": "Ada"},
                    {
                        "path": "/subject/name",
                        "op": "eq",
                        "value": "Ada Lovelace",
                    },
                ]
            },
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid"] is True
    assert body["root"] == committed["root"]
    assert "proofMode" not in body  # standard proof: no mode marker
    assert body["result"] == "pass"
    # One entry per condition, in policy traversal order, each echoing the
    # condition and its policy location.
    assert [entry["loc"] for entry in body["conditions"]] == [
        "policy.all[0]",
        "policy.all[1].any[0]",
        "policy.all[1].any[1]",
    ]
    assert _condition_results(body) == {
        "policy.all[0]": "pass",
        "policy.all[1].any[0]": "fail",
        "policy.all[1].any[1]": "pass",
    }
    first = body["conditions"][0]
    assert (first["path"], first["op"], first["value"]) == (
        "/subject/age",
        "gte",
        18,
    )


def test_gate_accepts_compact_proof_and_marks_mode():
    committed = _commit(proof_mode="compact")
    policy = {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 18},
            {"path": "/subject/isAdult", "op": "eq", "value": True},
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["proofMode"] == "compact"
    assert body["result"] == "pass"
    assert [entry["result"] for entry in body["conditions"]] == [
        "pass",
        "pass",
    ]


def test_gate_single_bare_condition_policy():
    committed = _commit()
    resp = _gate(
        committed, {"path": "/subject/age", "op": "gte", "value": 37}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"] == "fail"
    assert body["conditions"] == [
        {
            "loc": "policy",
            "path": "/subject/age",
            "op": "gte",
            "value": 37,
            "result": "fail",
        }
    ]


def test_gate_root_scalar_document_condition():
    committed = _commit(reveal=[""], document=42)
    resp = _gate(committed, {"path": "", "op": "gte", "value": 41})
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "pass"


# ---------------------------------------------------------------------------
# eq semantics: canonical JSON byte comparison
# ---------------------------------------------------------------------------


def test_gate_eq_compares_canonical_json_bytes():
    committed = _commit()
    policy = {
        "all": [
            # 36 and 36.0 canonicalize to different bytes -> fail.
            {"path": "/subject/age", "op": "eq", "value": 36.0},
            # Exact canonical match -> pass.
            {"path": "/subject/age", "op": "eq", "value": 36},
            # true and 1 are different canonical values -> fail.
            {"path": "/subject/isAdult", "op": "eq", "value": 1},
            {"path": "/subject/isAdult", "op": "eq", "value": True},
            # Empty containers compare by their canonical bytes too.
            {"path": "/emptyObj", "op": "eq", "value": {}},
            {"path": "/emptyObj", "op": "eq", "value": []},
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [entry["result"] for entry in body["conditions"]] == [
        "fail",
        "pass",
        "fail",
        "pass",
        "pass",
        "fail",
    ]
    assert body["result"] == "fail"


# ---------------------------------------------------------------------------
# gte semantics
# ---------------------------------------------------------------------------


def test_gate_gte_numeric_comparisons():
    committed = _commit()
    policy = {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 18},
            {"path": "/subject/age", "op": "gte", "value": 36},
            {"path": "/subject/age", "op": "gte", "value": 35.5},
            {"path": "/subject/age", "op": "gte", "value": 37},
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [entry["result"] for entry in body["conditions"]] == [
        "pass",
        "pass",
        "pass",
        "fail",
    ]
    assert body["result"] == "fail"


def test_gate_gte_non_numeric_disclosed_value_fails():
    committed = _commit()
    policy = {
        "all": [
            # A string leaf is not a number -> fail (not unknown).
            {"path": "/subject/name", "op": "gte", "value": 5},
            # A boolean leaf is not a number either.
            {"path": "/subject/isAdult", "op": "gte", "value": 0},
            # Nor is an empty container.
            {"path": "/emptyObj", "op": "gte", "value": 0},
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [entry["result"] for entry in body["conditions"]] == [
        "fail",
        "fail",
        "fail",
    ]
    assert body["result"] == "fail"


def test_gate_gte_threshold_must_be_non_boolean_finite_number():
    committed = _commit()
    for bad_value in (True, "18", None, [18], {"n": 18}):
        resp = _gate(
            committed, {"path": "/subject/age", "op": "gte", "value": bad_value}
        )
        assert resp.status_code == 400, (bad_value, resp.text)
        err = resp.json()["error"]
        assert err["code"] == "INVALID_REQUEST"
        assert err["loc"] == "policy.value"
        assert "non-boolean finite number" in err["message"]


# ---------------------------------------------------------------------------
# Three-valued logic
# ---------------------------------------------------------------------------


def test_gate_undisclosed_path_is_unknown_even_when_in_document():
    committed = _commit()  # /subject/id exists in DOC but is not disclosed
    resp = _gate(
        committed, {"path": "/subject/id", "op": "eq", "value": "u-123"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"] == "unknown"
    assert body["conditions"][0]["result"] == "unknown"


def test_gate_condition_paths_are_exact_not_patterns():
    committed = _commit()
    # "/emails/*" is a syntactically valid pointer but the gate does exact
    # leaf lookups: nothing was disclosed under that literal path.
    resp = _gate(committed, {"path": "/emails/*", "op": "eq", "value": "x"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "unknown"


def test_gate_all_false_dominates_unknown():
    committed = _commit()
    policy = {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 99},  # fail
            {"path": "/subject/id", "op": "eq", "value": "u-123"},  # unknown
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "fail"


def test_gate_all_unknown_when_no_failure():
    committed = _commit()
    policy = {
        "all": [
            {"path": "/subject/age", "op": "gte", "value": 18},  # pass
            {"path": "/subject/id", "op": "eq", "value": "u-123"},  # unknown
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "unknown"


def test_gate_any_true_dominates_unknown():
    committed = _commit()
    policy = {
        "any": [
            {"path": "/subject/age", "op": "gte", "value": 18},  # pass
            {"path": "/subject/id", "op": "eq", "value": "u-123"},  # unknown
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "pass"


def test_gate_any_unknown_when_no_success():
    committed = _commit()
    policy = {
        "any": [
            {"path": "/subject/age", "op": "gte", "value": 99},  # fail
            {"path": "/subject/id", "op": "eq", "value": "u-123"},  # unknown
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "unknown"


def test_gate_any_all_fail_is_fail_and_all_all_pass_is_pass():
    committed = _commit()
    resp = _gate(
        committed,
        {
            "any": [
                {"path": "/subject/age", "op": "gte", "value": 99},
                {"path": "/subject/name", "op": "eq", "value": "Bob"},
            ]
        },
    )
    assert resp.json()["result"] == "fail"

    resp = _gate(
        committed,
        {
            "all": [
                {"path": "/subject/age", "op": "gte", "value": 18},
                {"path": "/subject/name", "op": "eq", "value": "Ada Lovelace"},
            ]
        },
    )
    assert resp.json()["result"] == "pass"


def test_gate_deeply_nested_groups_propagate():
    committed = _commit()
    policy = {
        "any": [
            {
                "all": [
                    {"path": "/subject/age", "op": "gte", "value": 18},
                    {
                        "any": [
                            {
                                "path": "/subject/name",
                                "op": "eq",
                                "value": "Ada Lovelace",
                            }
                        ]
                    },
                ]
            },
            {"path": "/subject/id", "op": "eq", "value": "u-123"},
        ]
    }
    resp = _gate(committed, policy)
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"] == "pass"


# ---------------------------------------------------------------------------
# Policy structure errors (loc points into the policy)
# ---------------------------------------------------------------------------


def test_gate_missing_policy_field():
    committed = _commit()
    payload = {
        key: value for key, value in committed.items() if key != "proofMode"
    }
    resp = client.post("/gate/evaluate", json=payload)
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_REQUEST"
    assert err["loc"] == "policy"


def test_gate_policy_must_be_an_object():
    committed = _commit()
    resp = _gate(committed, [{"path": "/subject/age", "op": "eq", "value": 36}])
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_REQUEST"
    assert err["loc"] == "policy"


def test_gate_empty_groups_rejected_at_their_policy_loc():
    committed = _commit()
    resp = _gate(committed, {"all": []})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_REQUEST"
    assert err["loc"] == "policy.all"

    resp = _gate(
        committed,
        {
            "all": [
                {"path": "/subject/age", "op": "gte", "value": 18},
                {"any": []},
            ]
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy.all[1].any"


def test_gate_group_children_must_be_an_array():
    committed = _commit()
    resp = _gate(committed, {"all": {"path": "/a", "op": "eq", "value": 1}})
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy.all"


def test_gate_group_rejects_mixed_or_double_combinators():
    committed = _commit()
    resp = _gate(
        committed,
        {
            "all": [{"path": "/a", "op": "eq", "value": 1}],
            "any": [{"path": "/b", "op": "eq", "value": 2}],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy"

    resp = _gate(
        committed,
        {"all": [{"path": "/a", "op": "eq", "value": 1}], "path": "/a"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy"


def test_gate_non_object_policy_node_rejected():
    committed = _commit()
    resp = _gate(committed, {"all": ["not-a-node"]})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_REQUEST"
    assert err["loc"] == "policy.all[0]"


def test_gate_condition_field_errors_located():
    committed = _commit()

    # Missing fields.
    resp = _gate(committed, {"op": "eq", "value": 1})
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy.path"

    resp = _gate(committed, {"path": "/subject/age", "value": 1})
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy.op"

    resp = _gate(committed, {"path": "/subject/age", "op": "eq"})
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy.value"

    # Bad op.
    resp = _gate(
        committed, {"path": "/subject/age", "op": "lt", "value": 1}
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "INVALID_REQUEST"
    assert err["loc"] == "policy.op"

    # Non-string path.
    resp = _gate(committed, {"path": 7, "op": "eq", "value": 1})
    assert resp.status_code == 400
    assert resp.json()["error"]["loc"] == "policy.path"

    # Pointer syntax error, nested inside groups.
    resp = _gate(
        committed,
        {"any": [{"path": "/subject/a~x", "op": "eq", "value": 1}]},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "POINTER_SYNTAX"
    assert err["loc"] == "policy.any[0].path"


# ---------------------------------------------------------------------------
# Proof verification runs first and reuses the /verify errors
# ---------------------------------------------------------------------------


def test_gate_tampered_proof_reused_verify_error():
    committed = _commit()
    leaves = copy.deepcopy(committed["leaves"])
    leaves[0]["value"] = 999  # forge the first disclosed value
    resp = _gate(
        committed,
        {"path": "/subject/age", "op": "gte", "value": 18},
        leaves=leaves,
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "ROOT_MISMATCH"
    assert err["loc"] == "leaves[0]"


def test_gate_proof_error_wins_over_policy_error():
    committed = _commit()
    # Both the proof (tampered value) and the policy (empty group) are
    # broken; the proof is verified first, so its error is returned.
    leaves = copy.deepcopy(committed["leaves"])
    leaves[0]["value"] = 999
    resp = _gate(committed, {"all": []}, leaves=leaves)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "ROOT_MISMATCH"


def test_gate_structural_proof_errors_reused():
    committed = _commit()
    resp = _gate(
        committed,
        {"path": "/subject/age", "op": "gte", "value": 18},
        root="not-a-hash",
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "BAD_HASH_ENCODING"
    assert err["loc"] == "root"

    resp = _gate(
        committed,
        {"path": "/subject/age", "op": "gte", "value": 18},
        leaves=[],
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "EMPTY_REVEAL"


def test_gate_compact_proof_errors_reused():
    committed = _commit(proof_mode="compact")
    policy = {"path": "/subject/age", "op": "gte", "value": 18}

    # Structural compact error: drop a required proof node.
    nodes = copy.deepcopy(committed["proofNodes"])
    nodes.pop()
    resp = _gate(committed, policy, proofNodes=nodes)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "BAD_PROOF"

    # Cryptographic failure: tamper with a disclosed value.
    leaves = copy.deepcopy(committed["leaves"])
    leaves[0]["value"] = 999
    resp = _gate(committed, policy, leaves=leaves)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "ROOT_MISMATCH"


def test_gate_never_sees_the_document():
    # The request carries only root + disclosures + policy; undisclosed
    # facts stay unknown and there is no place for the original document.
    committed = _commit(reveal=["/subject/age"])
    resp = _gate(
        committed,
        {
            "all": [
                {"path": "/subject/age", "op": "gte", "value": 18},
                {"path": "/issuer", "op": "eq", "value": "https://id.example"},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"] == "unknown"
    assert "document" not in body
    assert "leaves" not in body
