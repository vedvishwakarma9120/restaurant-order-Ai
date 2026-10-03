import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

# ---------------- Config ----------------
MENU_FILE = "menu.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

RESTAURANT_INFO = """Restaurant: Bhukhkhad Cafe
Timings: 10 AM to 10 PM (Subah 10 baje se raat 10 baje tak)
Location: 123 Food Street, Downtown
Dine-in and takeout both available."""


# This exact line is always appended after a successful order confirmation,
# in English, no matter what language the rest of the reply is in.
CLOSING_LINE = "Thank you for ordering with Bhukhkhad Cafe!"

with open(MENU_FILE, "r", encoding="utf-8") as f:
    MENU = json.load(f)

COMMON_ALIASES = {
    "roti": "Tandoori Roti", "rotis": "Tandoori Roti", "chapati": "Tandoori Roti",
    "naan": "Butter Naan", "rice": "Steamed Basmati Rice", "chawal": "Steamed Basmati Rice",
    "chai": "Kulhad Chai", "tea": "Kulhad Chai", "coffee": "Cold Coffee",
    "maggi": "Masala Maggi", "fries": "Masala Fries", "dosa": "Masala Dosa",
    "momos": "Tandoori Momos", "burger": "Desi Paneer Burger", "pizza": "Butter Chicken Pizza",
    "noodles": "Veg Hakka Noodles", "wrap": "Peri Peri Paneer Wrap", "roll": "Paneer Kathi Roll",
}

ORDERS = {}  # order_id -> order dict

# Per-session menu tracking: {session_id: True/False}
# When user refreshes page / opens new tab, a new session_id is generated
# so they can request menu again.
MENU_SHOWN_SESSIONS = {}  # session_id -> bool
_CURRENT_SESSION_ID = ""   # Set per-request, used by tools

import difflib

# ---------------- Menu Search & Dish Resolution (Fast, Zero-RAM) ----------------
MODEL_READY = True


def _vector_search(query: str, k: int = 3):
    """Semantic & fuzzy search over the menu items by name, description, and keywords."""
    q = query.lower().strip()
    words = [w for w in q.split() if len(w) > 2]
    scored = []
    for item in MENU:
        name = item["dish_name"].lower()
        desc = item.get("description", "").lower()
        score = 0.0
        if q == name:
            score = 1.0
        elif q in name:
            score = 0.9
        elif any(w in name for w in words):
            score = 0.7
        elif q in desc or any(w in desc for w in words):
            score = 0.5
        else:
            ratio = difflib.SequenceMatcher(None, q, name).ratio()
            if ratio > 0.4:
                score = ratio * 0.8
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [item for score, item in scored[:k] if score > 0]
    return matches if matches else MENU[:k]


def _resolve_dish(raw_name: str, threshold: float = 0.45):
    """Exact -> alias -> substring -> fuzzy difflib matcher. Returns menu item or None."""
    q = raw_name.lower().strip()
    q = COMMON_ALIASES.get(q, q).lower()
    for item in MENU:
        if item["dish_name"].lower() == q:
            return item
    for item in MENU:
        if item["dish_name"].lower() in q or q in item["dish_name"].lower():
            return item
    best_item = None
    best_score = 0.0
    for item in MENU:
        name = item["dish_name"].lower()
        ratio = difflib.SequenceMatcher(None, q, name).ratio()
        if ratio > best_score:
            best_score = ratio
            best_item = item
    if best_score >= threshold:
        return best_item
    return None


def _alternatives(name: str, k: int = 2):
    hits = _vector_search(name, k=k + 3)
    return [h["dish_name"] for h in hits if h["available_quantity"] > 0][:k]


def _strip_trailing_thanks(text: str) -> str:
    """Strip any closing/thank-you line(s) the model may have generated on its own so we can
    replace them with the single fixed CLOSING_LINE, keeping the sign-off identical every time."""
    lines = text.rstrip().split("\n")
    while lines and re.search(r"thank|shukriya|dhanyavaad|dhanyawad", lines[-1], re.IGNORECASE):
        lines.pop()
    return "\n".join(lines).rstrip()


# ---------------- Rate Limiter ----------------
from collections import defaultdict

class RateLimiter:
    """In-memory sliding window rate limiter per client IP."""
    def __init__(self, max_requests: int = 20, window_seconds: int = 60, min_interval: float = 0.5):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_interval = min_interval
        self.requests = defaultdict(list)
        self.last_request = defaultdict(float)

    def is_allowed(self, client_id: str) -> tuple[bool, str]:
        now = time.time()
        # Burst check (minimum time between consecutive requests)
        last = self.last_request.get(client_id, 0.0)
        if (now - last) < self.min_interval:
            return False, "⚠️ Aap bohot jaldi messages bhej rahe hain. Kripya 1 second intezar karein."

        # Sliding window check
        cutoff = now - self.window_seconds
        timestamps = [t for t in self.requests[client_id] if t > cutoff]
        if len(timestamps) >= self.max_requests:
            oldest = timestamps[0]
            retry_after = max(1, int(self.window_seconds - (now - oldest)) + 1)
            return False, f"⚠️ Rate limit exceed ho gaya hai! (Max {self.max_requests} requests/min). Kripya {retry_after}s intezar karein."

        timestamps.append(now)
        self.requests[client_id] = timestamps
        self.last_request[client_id] = now
        return True, ""

RATE_LIMITER = RateLimiter(max_requests=15, window_seconds=60, min_interval=0.8)


# ---------------- Session state (with Attempt Limiter & Token Budget) ----------------
class Session:
    max_failed_attempts: int = 3
    lockout_duration: int = 600  # 10 minutes in seconds

    def __init__(self):
        self.pending = {}          # dish_key -> {"dish","quantity","price","prep_time"}
        self.active_order_id = None
        self.failed_attempts = 0   # Failed / cancelled order attempts counter
        self.blocked_until = 0.0   # Timestamp until which user is locked out (10 mins)
        self.total_tokens_used = 0 # Track session tokens for metrics

    def is_blocked(self) -> bool:
        if self.blocked_until > 0:
            if time.time() < self.blocked_until:
                return True
            else:
                self.blocked_until = 0.0
                self.failed_attempts = 0
                return False
        return False

    def get_lockout_message(self) -> str:
        rem_sec = max(0, int(self.blocked_until - time.time()))
        mins = rem_sec // 60
        secs = rem_sec % 60
        time_left = f"{mins} min {secs} sec" if mins > 0 else f"{secs} sec"
        return f"🚫 Security timeout active! Aapne 3 baar order fail/cancel kiya hai. Kripya {time_left} baad koshish karein."

    def record_failed_attempt(self) -> tuple[int, bool, str]:
        """Record an unconfirmed cancel or empty order attempt."""
        self.failed_attempts += 1
        if self.failed_attempts >= self.max_failed_attempts:
            self.blocked_until = time.time() + self.lockout_duration
            return self.max_failed_attempts, True, f"🚫 Security Lockout Activated! (Attempt {self.max_failed_attempts}/{self.max_failed_attempts}). 10-minute timeout shuru ho chuka hai."
        else:
            return self.failed_attempts, False, f"⚠️ Warning: Failed/Cancelled Attempt {self.failed_attempts}/{self.max_failed_attempts}. 3 attempts ke baad 10-minute security timeout lag jayega."

    def reset_attempts(self):
        self.failed_attempts = 0
        self.blocked_until = 0.0

    def reset_session(self):
        self.pending = {}
        self.active_order_id = None
        self.failed_attempts = 0
        self.blocked_until = 0.0


SESSION = Session()

# Tools

@tool
def search_menu_tool(query: str) -> str:
    """Fuzzy/semantic search over the menu for a dish name or craving. Use to disambiguate unclear
    dish names or to suggest options when nothing else matches."""
    matches = _vector_search(query, k=3)
    if not matches:
        return "No matching dishes found."
    return "; ".join(
        f"{m['dish_name']} (Rs.{m['price']}, {'available' if m['available_quantity'] > 0 else 'out of stock'})"
        for m in matches
    )


@tool
def get_full_menu_tool(force_show: bool = False) -> str:
    """Return the full menu as dish name and price only, one dish per line, short form,
    only items that are currently available. Use when the customer asks to see the menu.
    The menu is displayed once per chat session. If customer specifically asks to see it again (dobara/again/fir se),
    pass force_show=True."""
    global _CURRENT_SESSION_ID
    sid = _CURRENT_SESSION_ID
    if MENU_SHOWN_SESSIONS.get(sid, False) and not force_show:
        return (
            "NOTE: Menu has already been displayed once in this chat. Tell the customer: "
            "'Menu upar chat mein already share kiya gaya hai, aap scroll karke dekh sakte hain! "
            "Kisi specific dish ke baare mein jaanna ho ya order karna ho to batayein.'"
        )

    MENU_SHOWN_SESSIONS[sid] = True
    return "\n".join(
        f"{m['dish_name']} - Rs.{m['price']}"
        for m in MENU
        if m["available_quantity"] > 0
    )


@tool
def add_to_order_tool(dish_name: str, quantity: int = 1) -> str:
    """Resolve a dish name (handles typos/Hindi/Hinglish/aliases) and add it to the customer's pending
    order with the given quantity. Call this ONCE PER DISTINCT DISH when the customer lists several
    items in one message. Returns availability info to relay back to the customer."""
    item = _resolve_dish(dish_name)
    if not item:
        alts = _alternatives(dish_name)
        return f"'{dish_name}' not found on the menu. Alternatives: {', '.join(alts) or 'none'}."
    avail = item["available_quantity"]
    if avail == 0:
        alts = _alternatives(item["dish_name"])
        return f"'{item['dish_name']}' is OUT OF STOCK. Alternatives: {', '.join(alts) or 'none'}."
    qty = min(max(quantity, 1), avail)
    note = "" if qty == quantity else f" (only {avail} available, capped from {quantity})"
    key = item["dish_name"].lower()
    if key in SESSION.pending:
        SESSION.pending[key]["quantity"] = min(SESSION.pending[key]["quantity"] + qty, avail)
    else:
        SESSION.pending[key] = {
            "dish": item["dish_name"], "quantity": qty, "price": item["price"],
            "prep_time": item.get("preparation_time", 15),
        }
    return f"Added {qty}x {item['dish_name']} to order{note}."


@tool
def remove_from_order_tool(dish_name: str) -> str:
    """Remove a dish from the customer's pending order. Use when they say hatao/remove/nikal do/
    don't want X anymore."""
    item = _resolve_dish(dish_name)
    key = item["dish_name"].lower() if item else dish_name.lower()
    if key in SESSION.pending:
        removed = SESSION.pending.pop(key)
        return f"Removed {removed['dish']} from the order."
    for k in list(SESSION.pending.keys()):
        if key in k or k in key:
            removed = SESSION.pending.pop(k)
            return f"Removed {removed['dish']} from the order."
    return f"'{dish_name}' was not in the pending order."


@tool
def view_order_tool() -> str:
    """Show the current pending (unconfirmed) order with line items and total. Call this after any
    add/remove, right before asking the customer to confirm."""
    if not SESSION.pending:
        return "Pending order is empty."
    lines = [f"{v['quantity']}x {v['dish']} = Rs.{v['quantity'] * v['price']:.2f}" for v in SESSION.pending.values()]
    total = sum(v["quantity"] * v["price"] for v in SESSION.pending.values())
    return "Pending order:\n" + "\n".join(lines) + f"\nTotal: Rs.{total:.2f}"


@tool
def confirm_order_tool() -> str:
    """Finalize the pending order and send it to the kitchen. ONLY call this when the customer has
    explicitly agreed (haan/yes/confirm/ok/theek hai)."""
    if not SESSION.pending:
        _, is_blocked, warning = SESSION.record_failed_attempt()
        if is_blocked:
            return f"No pending order to confirm. {warning} {SESSION.get_lockout_message()}"
        return f"No pending order to confirm. {warning}"

    items = list(SESSION.pending.values())
    total = round(sum(i["price"] * i["quantity"] for i in items), 2)
    order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
    ORDERS[order_id] = {"items": items, "total": total, "status": "CONFIRMED", "created_at": datetime.now().isoformat()}
    SESSION.active_order_id = order_id
    SESSION.pending = {}
    SESSION.reset_attempts()  # Reset failed attempts counter on successful order

    ORDERS[order_id]["status"] = "COMPLETED"

    summary = ", ".join(f"{i['quantity']}x {i['dish']}" for i in items)
    return f"Order {order_id} confirmed, cooked and served. Items: {summary}. Total Rs.{total:.2f}."


@tool
def cancel_order_tool() -> str:
    """Cancel the pending (unconfirmed) order, or the last confirmed order if nothing is pending.
    Use when the customer says cancel/nahi/stop/mat karo."""
    _, is_blocked, warning = SESSION.record_failed_attempt()

    if SESSION.pending:
        SESSION.pending = {}
        return f"Pending order cleared. {warning}"
    elif SESSION.active_order_id and SESSION.active_order_id in ORDERS:
        oid = SESSION.active_order_id
        ORDERS[oid]["status"] = "CANCELLED"
        SESSION.active_order_id = None
        return f"Order {oid} cancelled. {warning}"
    else:
        return f"No order to cancel. {warning}"


@tool
def check_order_status_tool(order_id: str = "") -> str:
    """Check status of an order. Omit order_id to check the customer's most recent active order."""
    oid = order_id or SESSION.active_order_id
    if not oid or oid not in ORDERS:
        return "No matching order found."
    return f"Order {oid} status: {ORDERS[oid]['status']}"


TOOLS = [
    search_menu_tool, get_full_menu_tool, add_to_order_tool, remove_from_order_tool,
    view_order_tool, confirm_order_tool, cancel_order_tool, check_order_status_tool,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = f"""You are "Bhukhkhad Cafe", the official AI ordering assistant for Bhukhkhad Cafe.

{RESTAURANT_INFO}

CRITICAL RULE — ALWAYS OUTPUT TEXT:
- After calling ANY tool, you MUST ALWAYS write a customer-facing reply. Never return an empty response.
- If you called tools and got results, use those results to write your reply to the customer.
- Even if a tool result is an error or empty, still write a short polite reply.

STRICT OPERATING RULES:
1. CONCISENESS:
   - Keep replies short: 1-3 sentences max. No small talk, no filler, no emojis.
2. LANGUAGE CONSISTENCY:
   - Reply in the customer's language (English, Hindi, or Hinglish) from their LATEST message.
3. MENU DISPLAY (ONCE PER SESSION):
   - When customer asks for menu ("menu", "kya milega", "list", "show menu"), call get_full_menu_tool immediately.
   - If customer asks again ("dobara", "again", "fir se"), call get_full_menu_tool(force_show=True).
   - Output the complete dish list exactly as returned by the tool, one dish per line.
   - If tool says menu already shown, tell customer to scroll up in 1 sentence.
4. ORDERING FLOW:
   - When customer names items to order, call add_to_order_tool for EACH distinct item (one call per dish), then call view_order_tool.
   - After view_order_tool, show the order summary exactly as returned, then ask on a new line:
     English: "Shall I confirm this order?"
     Hindi/Hinglish: "Kya main yeh order confirm kar doon?"
   - For removals/changes: call remove_from_order_tool, then view_order_tool, ask confirmation again.
5. CONFIRMATION & CANCELLATION:
   - Customer says cancel/nahi/stop/mat karo/abort → call cancel_order_tool, then reply politely in 1 sentence.
   - Customer says haan/yes/confirm/ok/theek hai → call confirm_order_tool, then show Order ID + items + total.
6. ACCURACY:
   - NEVER invent dish names, prices, or inventory. Use tools only.
   - After successful confirm_order_tool: state Order ID, items, total. Do NOT write a thank-you (system adds it).
7. NO INTERNAL MONOLOGUE:
   - NEVER output your thoughts or planning. Output ONLY the clean customer-facing reply.   
"""

llm = (
    ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=2048,  # Generous limit to prevent any response truncation
    )
    if GROQ_API_KEY
    else None
)
llm_with_tools = llm.bind_tools(TOOLS) if llm else None


# ---------------- Agent loop ----------------
class RestaurantBot:
    def __init__(self):
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    def _trim(self, keep_turns: int = 3):
        """Keep system prompt and the last `keep_turns` Human-AI conversation turns safely."""
        human_idx = [i for i, m in enumerate(self.messages) if isinstance(m, HumanMessage)]
        if len(human_idx) > keep_turns:
            cut = human_idx[-keep_turns]
            self.messages = [self.messages[0]] + self.messages[cut:]

    def process_message(self, user_input: str, session_id: str = "") -> str:
        global _CURRENT_SESSION_ID
        _CURRENT_SESSION_ID = session_id

        # Quick reset command
        if user_input.strip().lower() in ["reset", "restart", "clear", "clear chat", "naya session"]:
            SESSION.reset_session()
            MENU_SHOWN_SESSIONS.pop(session_id, None)
            self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
            return "Session reset ho gaya hai! Namaste & Welcome to Bhukhkhad Cafe. Main aapki kya madad karoon?"

        # 1. Check if user is locked out
        if SESSION.is_blocked():
            return SESSION.get_lockout_message()

        # Direct fast menu response (shows once per chat session, per session_id)
        is_menu_req = bool(
            re.search(r"^\s*(?:please\s+)?(?:mujhe\s+)?(?:dobara\s+|fir\s*se\s+|phir\s*se\s+|again\s+)?(?:apna\s+)?(?:kripya\s+)?(?:menu|menu\s*card|list|kya\s+milega)(?:\s+dikhao|\s+bhejo|\s+show|\s+batao|\s+dekhna\s+hai)?\s*[.!?]?\s*$", user_input, re.I)
            or re.search(r"^\s*(?:show\s+)?(?:the\s+)?(?:menu)(?:\s+again|\s+please)?\s*[.!?]?\s*$", user_input, re.I)
        )
        if is_menu_req:
            force_show = bool(re.search(r"\b(dobara|again|fir se|phir se)\b", user_input, re.I))
            menu_already_shown = MENU_SHOWN_SESSIONS.get(session_id, False)
            if menu_already_shown and not force_show:
                return "📋 Menu upar chat mein already share kiya gaya hai, aap scroll karke dekh sakte hain! Kisi dish ke baare mein jaanna ho ya order karna ho to batayein."

            MENU_SHOWN_SESSIONS[session_id] = True
            menu_text = "\n".join(
                f"{m['dish_name']} - Rs.{m['price']}"
                for m in MENU
                if m["available_quantity"] > 0
            )
            return f"📋 Bhukhkhad Cafe Menu:\n\n{menu_text}\n\nAap kya order karna chahenge?"

        if not llm_with_tools:
            return "GROQ_API_KEY .env mein set nahi hai - kripya add karke restart karein."

        self.messages.append(HumanMessage(content=user_input))
        ai_msg = None
        order_confirmed_this_turn = False
        attempt_warning_captured = ""
        tokens_consumed = 0

        try:
            for _ in range(3):  # safety cap on tool-call rounds (token saver)
                ai_msg = llm_with_tools.invoke(self.messages)
                self.messages.append(ai_msg)

                if ai_msg and hasattr(ai_msg, "response_metadata"):
                    usage = ai_msg.response_metadata.get("token_usage", {})
                    tokens_consumed += usage.get("total_tokens", 0)

                if not ai_msg.tool_calls:
                    break
                for call in ai_msg.tool_calls:
                    fn = TOOLS_BY_NAME.get(call["name"])
                    try:
                        result = fn.invoke(call["args"]) if fn else f"Unknown tool: {call['name']}"
                    except Exception as e:
                        result = f"Tool error: {e}"

                    res_str = str(result)
                    if call["name"] == "confirm_order_tool" and res_str.startswith("Order "):
                        order_confirmed_this_turn = True

                    # Capture attempt / lockout warnings from tools
                    if "Attempt" in res_str or "Lockout" in res_str or "timeout" in res_str:
                        attempt_warning_captured = res_str

                    self.messages.append(ToolMessage(content=res_str, tool_call_id=call["id"]))
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate limit" in err_str.lower():
                return "Order service is busy. Kripya 5-10 second intezar karke dobara message karein."
            return "Maaf kijiye, server busy hai. Kripya thodi der baad dobara batayein."

        # Fallback check: If user clearly wanted to cancel, but LLM skipped calling cancel_order_tool
        is_cancel_intent = bool(re.search(r"\b(cancel|nahi chahiye|mat karo|abort)\b", user_input, re.I)) and not bool(re.search(r"\b(hatao|nikal|remove)\b", user_input, re.I))
        if is_cancel_intent and not attempt_warning_captured:
            tool_res = cancel_order_tool.invoke({})
            attempt_warning_captured = str(tool_res)

        # Fallback check: If user clearly wanted to confirm empty order, but LLM skipped confirm_order_tool
        is_confirm_intent = bool(re.search(r"\b(haan|yes|confirm|theek hai|ok|done)\b", user_input, re.I)) and not SESSION.pending and not bool(re.search(r"\b(nahi|not|cancel)\b", user_input, re.I))
        if is_confirm_intent and not attempt_warning_captured and not order_confirmed_this_turn:
            tool_res = confirm_order_tool.invoke({})
            attempt_warning_captured = str(tool_res)

        self._trim()
        reply = (ai_msg.content if ai_msg else "") or ""

        # Safety net: if model returned no text content after tool calls,
        # look at the last tool message for context and retry once with a clear nudge.
        if not reply.strip():
            # Find the last tool result to include as context for retry
            last_tool_content = ""
            for m in reversed(self.messages):
                if isinstance(m, ToolMessage):
                    last_tool_content = m.content
                    break

            retry_prompt = (
                f"You called tools and got this result: \"{last_tool_content}\". "
                "Now write a clear, friendly customer-facing reply based on this result. "
                "Do NOT call any more tools. Just reply to the customer directly."
            )
            self.messages.append(HumanMessage(content=retry_prompt))
            retry_msg = llm_with_tools.invoke(self.messages)
            self.messages.append(retry_msg)
            if retry_msg and hasattr(retry_msg, "response_metadata"):
                usage = retry_msg.response_metadata.get("token_usage", {})
                tokens_consumed += usage.get("total_tokens", 0)
            if retry_msg and retry_msg.content and retry_msg.content.strip():
                reply = retry_msg.content

        # Final fallback if still empty
        if not reply.strip():
            reply = "Maaf kijiye, kuch samajh nahi aaya. Dobara batayein?"

        # Clean any raw attempt/lockout fragments the LLM might have duplicated from history
        reply = re.sub(r"\(?Warning:\s*Failed/Cancelled\s*Attempt\s*\d+/\d+.*?\)?", "", reply, flags=re.IGNORECASE)
        reply = re.sub(r"\(?\d+\s*failed/cancelled\s*attempts\s*reached.*?\)?", "", reply, flags=re.IGNORECASE)
        reply = re.sub(r"🚫\s*Security\s*(?:timeout|Lockout).*?(?=\n|$)", "", reply, flags=re.IGNORECASE)
        reply = re.sub(r"⚠️\s*Aapne\s*3\s*baar.*?(?=\n|$)", "", reply, flags=re.IGNORECASE)

        # Clean raw backend/tool output that leaked into the customer reply
        # Remove duplicate "Order ORD-XXX confirmed..." lines (keep only the first)
        reply = re.sub(r"(Order ORD-[A-Z0-9]+ confirmed[^\n]*)(\n.*?Order ORD-[A-Z0-9]+ confirmed[^\n]*)+", r"\1", reply, flags=re.IGNORECASE)
        # Remove raw tool artifacts like "Added Xx ... to order", "Pending order:", tool status lines
        reply = re.sub(r"^Added \d+x .+ to order\.?\s*$", "", reply, flags=re.MULTILINE | re.IGNORECASE)
        reply = re.sub(r"^Pending order:.*$", "", reply, flags=re.MULTILINE | re.IGNORECASE)
        reply = re.sub(r"^No pending order to confirm\..*$", "", reply, flags=re.MULTILINE | re.IGNORECASE)
        reply = re.sub(r"^No order to cancel\..*$", "", reply, flags=re.MULTILINE | re.IGNORECASE)
        reply = re.sub(r"^No matching order found\..*$", "", reply, flags=re.MULTILINE | re.IGNORECASE)
        # Remove truncation notice if model included it
        reply = re.sub(r"\[The response was truncated.*?\]\.?", "", reply, flags=re.IGNORECASE)
        # Remove "Order ORD-XXX status: COMPLETED/CONFIRMED" raw tool lines (but keep friendly order confirmations)
        reply = re.sub(r"^Order ORD-[A-Z0-9]+ status:\s*\w+\.?\s*$", "", reply, flags=re.MULTILINE)
        # Collapse multiple blank lines
        reply = re.sub(r"\n{3,}", "\n\n", reply)
        reply = reply.strip()

        # Guarantee: If lockout was activated this turn or active, ensure user sees lockout message immediately
        if SESSION.is_blocked():
            lock_msg = SESSION.get_lockout_message()
            reply = f"{reply}\n\n{lock_msg}" if reply else lock_msg
        elif attempt_warning_captured:
            attempt_notice = f"⚠️ Warning: Failed/Cancelled Attempt {SESSION.failed_attempts}/{SESSION.max_failed_attempts}. 3 attempts ke baad 10-minute security timeout lag jayega."
            reply = f"{reply}\n\n{attempt_notice}" if reply else attempt_notice

        if order_confirmed_this_turn:
            reply = _strip_trailing_thanks(reply)
            reply = f"{reply}\n\n{CLOSING_LINE}"

        # Track session token consumption
        if tokens_consumed > 0:
            SESSION.total_tokens_used += tokens_consumed
        else:
            approx_tokens = (len(user_input) + len(reply)) // 3 + 120
            SESSION.total_tokens_used += approx_tokens

        return reply

    def process_message_stream(self, user_input: str, session_id: str = ""):
        full = self.process_message(user_input, session_id=session_id)
        words = full.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
            time.sleep(0.015)


# FastAPI app 
app = FastAPI(title="Bhukhkhad Cafe AI", version="1.0.0")
bot = RestaurantBot()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic request body schema
class ChatRequest(BaseModel):
    message: str = ""
    session_id: str = ""


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/index.html", methods=["GET", "HEAD"])
def index():
    return FileResponse("index.html", media_type="text/html")


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Uptime / Health endpoint."""
    return JSONResponse({
        "status": "ok",
        "menu_items": len(MENU),
        "failed_attempts": SESSION.failed_attempts,
        "is_blocked": SESSION.is_blocked(),
        "total_tokens_used": SESSION.total_tokens_used,
    })


@app.post("/reset")
def reset_endpoint():
    SESSION.reset_session()
    bot.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    return JSONResponse({"status": "reset", "reply": "Session reset ho gaya hai! Namaste & Welcome to Bhukhkhad Cafe."})


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    client_ip = get_client_ip(request)
    allowed, err_msg = RATE_LIMITER.is_allowed(client_ip)
    if not allowed:
        return JSONResponse({"reply": err_msg})

    try:
        reply = bot.process_message(body.message, session_id=body.session_id)
    except Exception as e:
        reply = f"Error: {e}"
    return JSONResponse({"reply": reply})


@app.post("/chat/stream")
def chat_stream(body: ChatRequest, request: Request):
    client_ip = get_client_ip(request)
    allowed, err_msg = RATE_LIMITER.is_allowed(client_ip)
    if not allowed:
        def rate_limited_stream():
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            rate_limited_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    def generate():
        try:
            for token in bot.process_message_stream(body.message, session_id=body.session_id):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except (BrokenPipeError, ConnectionResetError):
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def main():
    import uvicorn
    print("=== Bhukhkhad Cafe - LangChain Assistant (FastAPI) ===", flush=True)
    if not GROQ_API_KEY:
        print("[Warning] GROQ_API_KEY missing in .env - assistant will not respond until set.\n", flush=True)
    else:
        print(f"[Ready] Groq model: {GROQ_MODEL}\n", flush=True)

    port = int(os.environ.get("PORT", 10000))
    print(f"[Web UI] Listening on http://localhost:{port}\n", flush=True)
    uvicorn.run("robo:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()