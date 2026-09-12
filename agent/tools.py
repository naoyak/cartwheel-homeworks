"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from thefuzz import fuzz

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch
from seed.eligibility import effective_return_window_days, is_refund_eligible

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    ### YOUR CODE HERE (HW1)
    policies = [p for p in load_policy_docs() if p.policy_id == policy_id]
    if not policies:
        return {"ok": False, "error": "not_found", "reason": f"Policy {policy_id} not found."}
    policy = policies[0]
    return {
        "ok": True,
        "policy_id": policy.policy_id,
        "title": policy.title,
        "audience": policy.audience,
        "body": policy.body,
    }


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    query_tokens = [token.lower() for token in query.strip().split()]
    if not query_tokens:
        return {"ok": False, "error": "invalid_argument", "reason": "Query must be non-empty."}
    with db.connection() as conn:
        store_id = None
        if store:
            store_obj = db.get_store_by_name(conn, store)
            if not store_obj:
                return {"ok": False, "error": "not_found", "reason": f"Store '{store}' not found."}
            store_id = store_obj.id

        products = db.list_products(conn, store_id=store_id)
        filtered_products = []
        for product in products:
            title_desc = f"{product.title} {product.description}".lower() #.strip().split()
            if all(token in title_desc for token in query_tokens):
                if max_price_usd is None or product.price_usd <= max_price_usd:
                    filtered_products.append(product)

        filtered_products.sort(key=lambda p: (p.price_usd, p.id))
        limited_products = filtered_products[:max(1, min(limit, MAX_SEARCH_LIMIT))]

    return {
        "ok": True,
        "products": [
            {
                "product_id": p.id,
                "store_id": p.store_id,
                "title": p.title,
                "price_usd": p.price_usd,
            }
            for p in limited_products
        ],
        "count": len(limited_products),
    }


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "Support staff have no orders of their own; use get_order instead.",
        }
    with db.connection() as conn:
        if ctx.role == "shopper":
            order_objs = db.list_orders_for_user(conn, ctx.user_id)
        else: # ctx.role == "merchant":
            order_objs = db.list_orders_for_store(conn, ctx.store_id)

    return {
        "ok": True,
        "orders": [order.to_public_dict() for order in order_objs],
        "count": len(order_objs),
    }


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order_obj = db.get_order(conn, order_id)
        if order_obj is None:
            return {"ok": False, "error": "not_found", "reason": f"Order {order_id} not found."}
        if not can_cancel_order(ctx, order_obj.user_id, order_obj.store_id):
                    return permission_denied(f"{ctx.role} cannot cancel order {order_id}")
        if order_obj.status != "placed":
            return {
                "ok": False,
                "error": "not_eligible",
                "reason": f"Order {order_id} is in status '{order_obj.status}'; only 'placed' orders can be cancelled.",
            }

        db.set_order_status(conn, order_id, "cancelled")
        return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}.
    """
    fuzzy_match_threshold = 80
    with db.connection() as conn:
        if ctx.role == 'support':
            order_candidates = db.list_order_search_candidates(conn,all_orders=True)
        elif ctx.role == 'shopper':
            order_candidates = db.list_order_search_candidates(conn, user_id=ctx.user_id)
        elif ctx.role == 'merchant':
            order_candidates = db.list_order_search_candidates(conn, store_id=ctx.store_id)
        else:
            return {
                "ok": False,
                "error": "invalid_argument",
                "reason": f"Role '{ctx.role}' is not supported for order search.",
            }
        prod_name_map = {p.id: p.title for p in db.list_products(conn)} 
        order_candidates = [o for o in order_candidates if 
            fuzz.partial_ratio(query.lower(), prod_name_map[o.product_id].lower()) >= fuzzy_match_threshold]
        # order_candidates = [o.to_public_dict() for o in order_candidates[:5]]

    return {
        "ok": True,
        "orders": [o.to_public_dict() for o in order_candidates[:5]],
    }


def check_refund_eligibility(
    ctx: AuthContext, order_id: int, request_date: str | None = None
) -> dict[str, Any]:
    """Check whether an order is refund/return eligible. Risk tier: read.

    Custom HW1 tool. It surfaces, as structured data, the eligibility decision
    that otherwise only exists as the order's stamped ``refund_eligible`` flag
    or as prose in the policy documents. It determines eligibility from three
    things, in order:

    1. The order's status and delivery date (only a delivered order with a
       delivery date can be returned).
    2. The store's return-window override
       (agent.db.Store.return_window_days_override), which takes precedence.
    3. The platform default window (facts.yaml ``return_window_days``) when the
       store has no override.

    The window rule matches cw-store-overrides and the eligibility oracle in
    seed/eligibility.py, which this tool reuses rather than re-deriving.

    Access follows the order access matrix (agent.auth.can_view_order): a caller
    may only check an order in their own scope. An out-of-scope caller gets
    permission_denied and learns nothing about the order.

    Args:
        ctx: The caller's auth context.
        order_id: The order to check.
        request_date: Optional ISO date (YYYY-MM-DD) standing for when the
            refund/return is requested. Defaults to the world's current date
            (agent.db.world_asof), which is the date the order's stamped
            ``refund_eligible`` flag was computed against. Passing a different
            date answers a hypothetical ("would this be eligible if requested
            then?").

    Returns:
        On success: {"ok": True, "eligible": bool, "reason": <explanation>,
        "status", "delivered_at", "as_of", "effective_window_days",
        "window_source" ("store_override" or "platform_default"),
        "platform_window_days", "store_override_days",
        "stamped_refund_eligible"}.

        Errors: not_found for an unknown order; permission_denied for an order
        outside the caller's scope; invalid_argument for an unparseable
        request_date.
    """
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )
        if request_date is None:
            as_of = db.world_asof(conn)
        else:
            try:
                as_of = date.fromisoformat(request_date)
            except ValueError:
                return {
                    "ok": False,
                    "error": "invalid_argument",
                    "reason": (
                        "request_date must be an ISO date (YYYY-MM-DD); "
                        f"got {request_date!r}"
                    ),
                }

        store = db.get_store(conn, order.store_id)
        override = store.return_window_days_override if store else None
        platform_window = load_facts()["return_window_days"]
        window = effective_return_window_days(platform_window, override)
        window_source = "store_override" if override is not None else "platform_default"

        eligible = is_refund_eligible(
            status=order.status,
            delivered_at=order.delivered_at,
            as_of=as_of,
            return_window_days=window,
        )

        if order.status != "delivered":
            reason = (
                f"Order #{order_id} has status '{order.status}', so it is not "
                "refund-eligible; only delivered orders can be returned."
            )
        elif order.delivered_at is None:
            reason = (
                f"Order #{order_id} has no delivery date on record, so "
                "eligibility cannot be determined."
            )
        else:
            age = (as_of - order.delivered_at).days
            if override is not None and store is not None:
                window_label = f"{store.name}'s {window}-day store return window"
            else:
                window_label = f"the {window}-day platform return window"
            position = "within" if eligible else "past"
            reason = (
                f"Order #{order_id} was delivered {order.delivered_at.isoformat()} "
                f"({age} days before {as_of.isoformat()}), {position} {window_label}, "
                f"so it is {'refund-eligible' if eligible else 'not refund-eligible'}."
            )

        return {
            "ok": True,
            "order_id": order_id,
            "eligible": eligible,
            "reason": reason,
            "status": order.status,
            "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
            "as_of": as_of.isoformat(),
            "effective_window_days": window,
            "window_source": window_source,
            "platform_window_days": platform_window,
            "store_override_days": override,
            "stamped_refund_eligible": order.refund_eligible,
        }