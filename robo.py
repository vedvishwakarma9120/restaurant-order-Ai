import json
import os
import re
import sys
import time
import uuid
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

# ---------------- Config ----------------
MENU_FILE = "menu.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # tool-calling capable

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

# ---------------- RAG (Qdrant + embeddings) ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(":memory:")
qdrant.create_collection("menu", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
qdrant.upsert(
    collection_name="menu",
    points=[
        PointStruct(
            id=item["id"],
            vector=embedder.encode(f"{item['dish_name']}: {item['description']}").tolist(),
            payload=item,
        )
        for item in MENU
    ],
)


def _vector_search(query: str, k: int = 3):
    vec = embedder.encode(query).tolist()
    return [h.payload for h in qdrant.query_points(collection_name="menu", query=vec, limit=k).points]


def _resolve_dish(raw_name: str, threshold: float = 0.42):
    """Exact -> alias -> substring -> vector-search-with-confidence-threshold. Returns menu item or None."""
    q = raw_name.lower().strip()
    q = COMMON_ALIASES.get(q, q).lower()
    for item in MENU:
        if item["dish_name"].lower() == q:
            return item
    for item in MENU:
        if item["dish_name"].lower() in q or q in item["dish_name"].lower():
            return item
    vec = embedder.encode(raw_name).tolist()
    hits = qdrant.query_points(collection_name="menu", query=vec, limit=1).points
    if hits and hits[0].score >= threshold:  # <-- confidence gate (fixes random-match bug)
        return hits[0].payload
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


# ---------------- Session state (per-bot; simple single-user like original) ----------------
class Session:
    def __init__(self):
        self.pending = {}          # dish_key -> {"dish","quantity","price","prep_time"}
        self.active_order_id = None


SESSION = Session()

# ---------------- Tools (this replaces the old regex/JSON-NLU + if/else pipeline) ----------------

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
def get_full_menu_tool() -> str:
    """Return the full menu as dish name and price only, one dish per line, short form,
    only items that are currently available. Use when the customer asks to see the menu.
    Relay this output to the customer exactly as-is, one line per dish — do not summarize it
    into a sentence and do not add description or availability text."""
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
        return "No pending order to confirm."
    items = list(SESSION.pending.values())
    total = round(sum(i["price"] * i["quantity"] for i in items), 2)
    order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
    ORDERS[order_id] = {"items": items, "total": total, "status": "CONFIRMED", "created_at": datetime.now().isoformat()}
    SESSION.active_order_id = order_id
    SESSION.pending = {}

    ORDERS[order_id]["status"] = "COOKING"
    max_prep = max((i.get("prep_time", 15) for i in items), default=15)
    time.sleep(min(max(max_prep // 5, 2), 4))
    ORDERS[order_id]["status"] = "COMPLETED"

    summary = ", ".join(f"{i['quantity']}x {i['dish']}" for i in items)
    return f"Order {order_id} confirmed, cooked and served. Items: {summary}. Total Rs.{total:.2f}."


@tool
def cancel_order_tool() -> str:
    """Cancel the pending (unconfirmed) order, or the last confirmed order if nothing is pending.
    Use when the customer says cancel/nahi/stop/mat karo."""
    if SESSION.pending:
        SESSION.pending = {}
        return "Pending order cleared."
    if SESSION.active_order_id and SESSION.active_order_id in ORDERS:
        oid = SESSION.active_order_id
        ORDERS[oid]["status"] = "CANCELLED"
        SESSION.active_order_id = None
        return f"Order {oid} cancelled."
    return "No order to cancel."


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

SYSTEM_PROMPT = f"""You are "Bhukhkhad Cafe", the ordering assistant for Bhukhkhad Cafe.

{RESTAURANT_INFO}

Rules:
- Detect the customer's language (English, Hindi, or Hinglish) from their LATEST message and reply in that same style.
- NEVER invent dish names, prices or availability yourself — always use the tools for real data.
- If the customer asks to see the menu, call get_full_menu_tool and paste its output back to the
  customer EXACTLY as returned, one dish per line ("Dish - Rs.price"). Do NOT summarize it into a
  sentence, do NOT add description/availability text, and do NOT say things like "here is our menu"
  followed by nothing — the actual line-by-line list must appear in your reply.
- If the customer names one or more dishes to order, call add_to_order_tool ONCE PER DISTINCT DISH,
  then call view_order_tool, then show the exact order summary text returned by view_order_tool
  (do not reword the item/total lines), followed on a new line by ONE of these exact confirmation
  questions, matching the customer's language/style — do not invent any other wording:
    English: "Shall I confirm this order?"
    Hindi/Hinglish: "Kya main yeh order confirm kar doon?"
- If the customer wants to remove/change an item, call remove_from_order_tool, then view_order_tool,
  then show the updated summary (again exactly as returned) and ask again using the same fixed
  confirmation question above.
- Only call confirm_order_tool on a clear yes/haan/confirm. Only call cancel_order_tool on a clear no/nahi/cancel.
- If a dish is unavailable or not found, relay the alternatives the tool gives you.
- Mention timings/location ONLY if the customer asks about them.
- After confirm_order_tool succeeds, show the Order ID, items and total in the customer's language/style,
  but do NOT write your own thank-you / closing line — the system appends a fixed one automatically.
- No emojis. Keep replies short, warm and natural.
"""

llm = (
    ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,

        reasoning_format="hidden",  # gpt-oss is a reasoning model — this stops its internal
                                     # "thinking" text from leaking into the customer-facing reply.
    )
    if GROQ_API_KEY
    else None
)
llm_with_tools = llm.bind_tools(TOOLS) if llm else None


# ---------------- Agent loop (replaces the old giant if/else state machine) ----------------
class RestaurantBot:
    def __init__(self):
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    def _trim(self, keep_turns: int = 8):
        human_idx = [i for i, m in enumerate(self.messages) if isinstance(m, HumanMessage)]
        if len(human_idx) > keep_turns:
            cut = human_idx[-keep_turns]
            self.messages = [self.messages[0]] + self.messages[cut:]

    def process_message(self, user_input: str) -> str:
        if not llm_with_tools:
            return "GROQ_API_KEY .env mein set nahi hai — kripya add karke restart karein."

        self.messages.append(HumanMessage(content=user_input))
        ai_msg = None
        order_confirmed_this_turn = False
        for _ in range(6):  # safety cap on tool-call rounds
            ai_msg = llm_with_tools.invoke(self.messages)
            self.messages.append(ai_msg)
            if not ai_msg.tool_calls:
                break
            for call in ai_msg.tool_calls:
                fn = TOOLS_BY_NAME.get(call["name"])
                try:
                    result = fn.invoke(call["args"]) if fn else f"Unknown tool: {call['name']}"
                except Exception as e:
                    result = f"Tool error: {e}"
                if call["name"] == "confirm_order_tool" and isinstance(result, str) and result.startswith("Order "):
                    order_confirmed_this_turn = True
                self.messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        self._trim()
        reply = (ai_msg.content if ai_msg else "") or "Maaf kijiye, kuch samajh nahi aaya. Dobara batayein?"

        # Safety net: a suspiciously short/garbled reply (model sometimes mangles Hinglish text)
        # gets one retry with a stricter nudge instead of reaching the customer as-is.
        if len(reply.strip()) < 8 or not re.search(r"[a-zA-Z]{3,}", reply):
            self.messages.append(HumanMessage(
                content="(system note: your last reply looked incomplete or garbled — "
                        "please resend a clear, well-formed reply in the same language/style.)"
            ))
            retry_msg = llm_with_tools.invoke(self.messages)
            self.messages.append(retry_msg)
            if retry_msg.content and len(retry_msg.content.strip()) >= 8:
                reply = retry_msg.content

        if order_confirmed_this_turn:
            reply = _strip_trailing_thanks(reply)
            reply = f"{reply}\n\n{CLOSING_LINE}"

        return reply

    def process_message_stream(self, user_input: str):
        full = self.process_message(user_input)
        words = full.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
            time.sleep(0.015)


# ---------------- HTTP server for the existing frontend ----------------
class ChatHandler(BaseHTTPRequestHandler):
    bot_instance = None

    def log_message(self, format, *args):
        return

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors()
            self.end_headers()
            with open("index.html", "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    def do_POST(self):
        if self.path == "/chat/stream":
            length = int(self.headers.get("Content-Length", 0))
            try:
                user_msg = json.loads(self.rfile.read(length).decode("utf-8")).get("message", "")
            except Exception:
                user_msg = ""

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            try:
                for token in self.bot_instance.process_message_stream(user_msg):
                    self.wfile.write(f"data: {json.dumps({'token': token})}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.close_connection = True

        elif self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            try:
                user_msg = json.loads(self.rfile.read(length).decode("utf-8")).get("message", "")
                reply = self.bot_instance.process_message(user_msg)
            except Exception as e:
                reply = f"Error: {e}"
            body = json.dumps({"reply": reply}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()


def main():
    print("=== Bhukhkhad Cafe — LangChain Assistant ===")
    if not GROQ_API_KEY:
        print("[Warning] GROQ_API_KEY missing in .env — assistant will not respond until set.\n")
    else:
        print(f"[Ready] Groq model: {GROQ_MODEL}\n")

    bot = RestaurantBot()
    ChatHandler.bot_instance = bot

    server = None
    for port in [5000, 5001, 8080]:
        try:
            server = ThreadingHTTPServer(("0.0.0.0", port), ChatHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"[Web UI] http://localhost:{port}\n")
            break
        except Exception:
            continue
    if not server:
        print("[Web UI] Could not bind to 5000/5001/8080. Terminal chat still works.")

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Bot: Thank you for visiting! Have a wonderful day.")
                break
            print("Bot: ", end="", flush=True)
            for token in bot.process_message_stream(user_input):
                sys.stdout.write(token)
                sys.stdout.flush()
            print("\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()