"""Policy gate evaluation over verified disclosures.

A *policy* is a tree of ``all`` / ``any`` groups down to leaf conditions
``{"path", "op", "value"}``:

* ``path`` — an exact RFC 6901 leaf path, resolved against the **verified
  disclosures only**; a path that was not disclosed evaluates to
  ``unknown`` (the gate never sees the original document);
* ``op`` — ``"eq"`` compares the disclosed value with the constant by
  canonical JSON bytes (the same canonical form used for leaf hashing, so
  ``1`` and ``1.0`` differ, as do ``true`` and ``1``); ``"gte"`` compares
  numerically and only accepts a non-boolean finite number as its
  threshold — a disclosed value that is not a non-boolean number
  evaluates to ``fail``;
* ``value`` — the comparison constant (any JSON value for ``eq``).

Groups fold with three-valued logic: ``all`` is ``fail`` as soon as one
child fails (otherwise ``unknown`` when any child is unknown), ``any`` is
``pass`` as soon as one child passes (otherwise ``unknown`` when any
child is unknown).  The overall verdict is ``pass`` / ``fail`` /
``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, Union

from . import pointers
from .canonical import canonical_json_bytes
from .errors import ApiError, INVALID_REQUEST, POINTER_SYNTAX
from .validation import validate_json_value

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"

_OPS = ("eq", "gte")
_CONDITION_KEYS = ("path", "op", "value")


@dataclass(frozen=True)
class Condition:
    """One leaf condition: compare the disclosed value at ``path``."""

    loc: str  # location inside the policy tree, e.g. "policy.all[1].any[0]"
    path: str
    op: str
    value: Any


@dataclass(frozen=True)
class Group:
    """An ``all`` / ``any`` combinator over a non-empty child list."""

    combinator: str
    children: Tuple["Node", ...]


Node = Union[Group, Condition]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def parse_policy(raw: Any) -> Node:
    """Validate the policy tree and return its parsed form.

    Every structural problem raises :class:`ApiError` with a ``loc``
    pointing at the offending policy node (``policy``, ``policy.all[1]``,
    ``policy.all[1].any[0].op`` ...).  Empty groups and malformed
    conditions are rejected before any evaluation happens.
    """
    return _parse_node(raw, "policy")


def _parse_node(raw: Any, loc: str) -> Node:
    if not isinstance(raw, dict):
        raise ApiError(
            INVALID_REQUEST,
            "policy node must be an object: an 'all'/'any' group or a "
            "{path, op, value} condition",
            loc=loc,
        )

    combinators = [key for key in ("all", "any") if key in raw]
    if combinators:
        if len(combinators) > 1:
            raise ApiError(
                INVALID_REQUEST,
                "a policy group carries exactly one of 'all' / 'any', "
                "got both",
                loc=loc,
            )
        if any(key in raw for key in _CONDITION_KEYS):
            raise ApiError(
                INVALID_REQUEST,
                "a policy group must not also carry condition fields "
                "('path' / 'op' / 'value')",
                loc=loc,
            )
        combinator = combinators[0]
        group_loc = f"{loc}.{combinator}"
        children_raw = raw[combinator]
        if not isinstance(children_raw, list):
            raise ApiError(
                INVALID_REQUEST,
                f"'{combinator}' must be an array of policy nodes, got "
                f"{type(children_raw).__name__}",
                loc=group_loc,
            )
        if len(children_raw) == 0:
            raise ApiError(
                INVALID_REQUEST,
                f"'{combinator}' must contain at least one policy node; "
                "empty groups are rejected",
                loc=group_loc,
            )
        return Group(
            combinator,
            tuple(
                _parse_node(child, f"{group_loc}[{i}]")
                for i, child in enumerate(children_raw)
            ),
        )

    # Leaf condition ------------------------------------------------------
    if "path" not in raw:
        raise ApiError(
            INVALID_REQUEST, "missing required field 'path'", loc=f"{loc}.path"
        )
    path = raw["path"]
    if not isinstance(path, str):
        raise ApiError(
            INVALID_REQUEST,
            f"'path' must be a pointer string, got {type(path).__name__}",
            loc=f"{loc}.path",
        )
    try:
        pointers.parse_pointer(path)
    except pointers.PointerError as exc:
        raise ApiError(POINTER_SYNTAX, str(exc), loc=f"{loc}.path") from exc

    if "op" not in raw:
        raise ApiError(
            INVALID_REQUEST, "missing required field 'op'", loc=f"{loc}.op"
        )
    op = raw["op"]
    if op not in _OPS:
        raise ApiError(
            INVALID_REQUEST,
            f"'op' must be 'eq' or 'gte', got {op!r}",
            loc=f"{loc}.op",
        )

    if "value" not in raw:
        raise ApiError(
            INVALID_REQUEST,
            "missing required field 'value'",
            loc=f"{loc}.value",
        )
    value = raw["value"]
    validate_json_value(value, loc=f"{loc}.value")
    if op == "gte" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        # Finiteness is already enforced by validate_json_value above.
        raise ApiError(
            INVALID_REQUEST,
            f"'gte' threshold must be a non-boolean finite number, got "
            f"{value!r}",
            loc=f"{loc}.value",
        )
    return Condition(loc=loc, path=path, op=op, value=value)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    node: Node, values_by_path: Mapping[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Fold *node* over the verified disclosures.

    *values_by_path* maps each verified leaf path to its value.  Returns
    the overall ``pass`` / ``fail`` / ``unknown`` verdict plus one result
    entry per condition, in policy traversal order.
    """
    conditions: List[Dict[str, Any]] = []

    def walk(current: Node) -> str:
        if isinstance(current, Condition):
            result = _eval_condition(current, values_by_path)
            conditions.append(
                {
                    "loc": current.loc,
                    "path": current.path,
                    "op": current.op,
                    "value": current.value,
                    "result": result,
                }
            )
            return result
        results = [walk(child) for child in current.children]
        if current.combinator == "all":
            if FAIL in results:
                return FAIL
            if UNKNOWN in results:
                return UNKNOWN
            return PASS
        if PASS in results:
            return PASS
        if UNKNOWN in results:
            return UNKNOWN
        return FAIL

    return walk(node), conditions


def _eval_condition(cond: Condition, values_by_path: Mapping[str, Any]) -> str:
    if cond.path not in values_by_path:
        return UNKNOWN
    disclosed = values_by_path[cond.path]
    if cond.op == "eq":
        same = canonical_json_bytes(disclosed) == canonical_json_bytes(
            cond.value
        )
        return PASS if same else FAIL
    # gte: a disclosed value that is not a non-boolean number fails.
    if isinstance(disclosed, bool) or not isinstance(disclosed, (int, float)):
        return FAIL
    return PASS if disclosed >= cond.value else FAIL
