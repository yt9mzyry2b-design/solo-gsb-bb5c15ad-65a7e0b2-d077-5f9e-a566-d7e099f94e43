"""Policy gate engine for ``POST /gate/evaluate``.

A policy is a recursively nested tree made of two node kinds:

* group node: ``{"all": [node, ...]}`` or ``{"any": [node, ...]}`` with a
  non-empty children array (the two group keys are mutually exclusive and
  cannot be mixed with leaf-condition fields);
* leaf condition: ``{"path": "<RFC 6901 pointer>", "op": "eq"|"gte",
  "value": <threshold>}``.

Values are read **only** from the map of verified disclosed leaves passed to
:func:`evaluate_policy`; the caller is responsible for proving the
disclosure against the committed root before invoking this module. A leaf
condition whose path was not disclosed evaluates to ``unknown``; groups then
follow three-valued logic:

* ``all`` — a ``fail`` child makes the group fail even when other children
  are unknown; otherwise an ``unknown`` child propagates; the group passes
  only when every child passes;
* ``any`` — a ``pass`` child makes the group pass even when other children
  are unknown; otherwise an ``unknown`` child propagates; the group fails
  only when every child fails.

Condition semantics:

* ``eq`` compares the canonical JSON bytes (the same canonical form used by
  ``/commit`` and ``/diff``) of the disclosed value and the threshold, so
  ``1`` and ``1.0`` or ``true`` and ``1`` are different;
* ``gte`` accepts only non-boolean finite numbers as thresholds; the
  comparison is false (never an error) when the disclosed value is not a
  finite number — a string, boolean, null or empty container.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from . import pointers
from .canonical import canonical_json_bytes
from .errors import ApiError, INVALID_REQUEST, POINTER_SYNTAX
from .validation import validate_json_value

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"

_GROUP_KEYS = ("all", "any")
_CONDITION_FIELDS = ("path", "op", "value")


def _invalid(
    message: str,
    loc: str,
    *,
    code: str = INVALID_REQUEST,
    detail: Dict[str, Any] | None = None,
) -> None:
    raise ApiError(code, message, loc=loc, detail=detail)


def validate_policy(node: Any, loc: str = "policy") -> None:
    """Structurally validate *node*, raising :class:`ApiError` on any fault.

    ``loc`` is the dotted/Indexed path of *node* inside the request, e.g.
    ``policy.all[2].any[0].op``; every error therefore points at the exact
    policy node or field that is invalid.
    """
    if not isinstance(node, dict):
        _invalid(
            "policy node must be an object: a group {'all': [...]} / "
            "{'any': [...]} or a leaf condition {'path', 'op', 'value'}, "
            f"got {type(node).__name__}",
            loc,
        )

    keys = set(node)
    group_keys = [key for key in _GROUP_KEYS if key in node]
    if len(group_keys) > 1:
        _invalid(
            "a policy group must carry exactly one of 'all'/'any'; the two "
            "are mutually exclusive",
            f"{loc}.all",
        )

    if group_keys:
        group_key = group_keys[0]
        mixed = [key for key in _CONDITION_FIELDS if key in keys]
        if mixed:
            _invalid(
                f"leaf-condition field {mixed[0]!r} cannot appear on a "
                f"'{group_key}' group node",
                f"{loc}.{mixed[0]}",
            )
        extra = sorted(keys - {group_key})
        if extra:
            _invalid(
                f"unknown field {extra[0]!r} on a policy group node",
                f"{loc}.{extra[0]}",
            )

        children = node[group_key]
        if not isinstance(children, list):
            _invalid(
                f"'{group_key}' must be an array of policy nodes, got "
                f"{type(children).__name__}",
                f"{loc}.{group_key}",
            )
        if len(children) == 0:
            _invalid(
                f"'{group_key}' group must contain at least one node; empty "
                "groups are not allowed",
                f"{loc}.{group_key}",
            )
        for i, child in enumerate(children):
            validate_policy(child, f"{loc}.{group_key}[{i}]")
        return

    # Leaf condition node.
    for field in _CONDITION_FIELDS:
        if field not in node:
            _invalid(
                f"leaf condition is missing required field {field!r}",
                f"{loc}.{field}",
            )
    extra = sorted(keys - set(_CONDITION_FIELDS))
    if extra:
        _invalid(
            f"unknown field {extra[0]!r} on a leaf condition; allowed "
            "fields are 'path', 'op' and 'value'",
            f"{loc}.{extra[0]}",
        )

    path = node["path"]
    if not isinstance(path, str):
        _invalid(
            f"'path' must be an RFC 6901 pointer string, got "
            f"{type(path).__name__}",
            f"{loc}.path",
        )
    try:
        pointers.parse_pointer(path)
    except pointers.PointerError as exc:
        _invalid(str(exc), f"{loc}.path", code=POINTER_SYNTAX)

    op = node["op"]
    if op not in ("eq", "gte"):
        _invalid(
            f"'op' must be 'eq' or 'gte', got {op!r}",
            f"{loc}.op",
        )

    threshold = node["value"]
    # Same finite/encodable-JSON discipline as every other endpoint; NaN,
    # Infinity and lone surrogates are rejected here with UNSUPPORTED_VALUE.
    validate_json_value(threshold, loc=f"{loc}.value")
    if op == "gte" and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
    ):
        _invalid(
            "'gte' threshold must be a finite JSON number, not a boolean, "
            f"string or {type(threshold).__name__}",
            f"{loc}.value",
        )


def _condition_result(
    condition: Dict[str, Any], disclosed: Dict[str, Any]
) -> str:
    path = condition["path"]
    if path not in disclosed:
        # The leaf exists in the committed tree but was not disclosed; the
        # verifier has no value to read.
        return UNKNOWN

    value = disclosed[path]
    op = condition["op"]
    threshold = condition["value"]

    if op == "eq":
        return (
            PASS
            if canonical_json_bytes(value) == canonical_json_bytes(threshold)
            else FAIL
        )

    # gte: a non-numeric disclosed value (string, boolean, null or empty
    # container) is simply false, never an error. Booleans are rejected as
    # numbers explicitly even though bool subclasses int in Python.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return FAIL
    if isinstance(value, float) and not math.isfinite(value):
        # Unreachable through the HTTP API (verified leaf values are already
        # finite), kept so direct callers get the documented false verdict.
        return FAIL
    return PASS if value >= threshold else FAIL


def _combine(op: str, results: List[str]) -> str:
    if op == "all":
        if FAIL in results:
            return FAIL
        if UNKNOWN in results:
            return UNKNOWN
        return PASS
    # any: a pass dominates, otherwise unknown propagates, else all fail.
    if PASS in results:
        return PASS
    if UNKNOWN in results:
        return UNKNOWN
    return FAIL


def evaluate_policy(
    node: Dict[str, Any], disclosed: Dict[str, Any]
) -> Tuple[str, List[Dict[str, str]]]:
    """Evaluate a validated policy against verified *disclosed* leaves.

    Returns ``(overall, conditions)`` where ``overall`` is one of
    pass/fail/unknown and ``conditions`` lists every leaf condition in
    depth-first policy order with its own pass/fail/unknown result.
    """
    conditions: List[Dict[str, str]] = []

    def visit(current: Dict[str, Any]) -> str:
        for group_key in _GROUP_KEYS:
            if group_key in current:
                results = [visit(child) for child in current[group_key]]
                return _combine(group_key, results)

        result = _condition_result(current, disclosed)
        conditions.append(
            {"path": current["path"], "op": current["op"], "result": result}
        )
        return result

    return visit(node), conditions
