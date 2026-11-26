import time
from pathlib import Path

from django.contrib.auth.models import AnonymousUser
from django.core.management.base import BaseCommand, CommandError

from accounts.models import Customer
from support.graph import RECURSION_LIMIT, build_graph, initial_state
from support.llm import LLMUnavailable


def parse_line(line, default_email):
    """'+ প্রশ্ন' = আগের কথোপকথনের ফলো-আপ; '@email প্রশ্ন' = ওই customer; '@none প্রশ্ন' = লগইন ছাড়া।
    ফেরত: (email, text, follow_up)"""
    if line.startswith("+"):
        return None, line[1:].strip(), True
    if line.startswith("@"):
        who, _, text = line[1:].partition(" ")
        return (None if who.lower() == "none" else who), text.strip(), False
    return default_email, line.strip(), False


class Command(BaseCommand):
    help = "Run questions through the agent graph and show every step. Development tool only."

    def add_arguments(self, parser):
        parser.add_argument("message", nargs="?", default=None)
        parser.add_argument("--as", dest="as_email", default=None,
                            help="Act as this customer's email. Without it: anonymous.")
        parser.add_argument("--file", default=None,
                            help="UTF-8 file. One question per line. Prefix '@email ' or '@none ' "
                                 "to choose the customer, '+ ' for a follow-up. '#' = comment.")
        parser.add_argument("--pause", type=float, default=0.0,
                            help="Seconds to wait between questions (rate limits).")

    def handle(self, *args, **options):
        items = []
        if options["file"]:
            path = Path(options["file"])
            if not path.is_file():
                raise CommandError(f"File not found: {path}")
            for raw in path.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    items.append(parse_line(line, options["as_email"]))
        elif options["message"]:
            items.append((options["as_email"], options["message"].strip(), False))
        else:
            raise CommandError("Give a message or --file.")

        customers = {}
        for email, _, _ in items:
            if email and email not in customers:
                try:
                    customers[email] = Customer.objects.get(email=email)
                except Customer.DoesNotExist:
                    raise CommandError(f"No customer with email {email}")

        prev_email, history = None, []
        for n, (email, text, follow_up) in enumerate(items, 1):
            if follow_up:
                if n == 1:
                    raise CommandError("A '+' line needs an earlier question.")
                email = prev_email
            else:
                history = []
            if n > 1 and options["pause"]:
                time.sleep(options["pause"])
            user = customers.get(email) if email else None
            answer = self._run_one(n, user, text, history)
            if answer is not None:
                history = history + [("user", text), ("assistant", answer)]
            prev_email = email

    def _run_one(self, n, user, text, history):
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING(
            f"\n=== {n}. acting as: {user.email if user else 'anonymous'}"
            f"{'  (follow-up)' if history else ''}"))
        w(f"question: {text}")
        if text.isascii() and text.count("?") >= 3:
            w(self.style.WARNING("WARNING: this looks like '???'. Your terminal probably "
                                 "destroyed the Bangla text. Use --file with a UTF-8 file."))

        graph = build_graph(user or AnonymousUser())
        tokens_in = tokens_out = 0
        answer = None
        try:
            for step in graph.stream(initial_state(history, text),
                                     config={"recursion_limit": RECURSION_LIMIT},
                                     stream_mode="updates"):
                for node, update in step.items():
                    update = update or {}
                    tokens_in += update.get("input_tokens", 0)
                    tokens_out += update.get("output_tokens", 0)
                    self._show(node, update)
                    if node == "finish":
                        answer = update["answer"]
        except LLMUnavailable:
            w(self.style.ERROR("LLM call failed. Check the log above."))
            return None

        w(f"\n{answer}\n")
        w(self.style.SUCCESS(f"[in={tokens_in} out={tokens_out}]"))
        return answer

    def _show(self, node, update):
        w = self.stdout.write
        if node == "understand":
            w(f"  [understand] intent={update.get('intent')} | standalone: "
              f"{update.get('standalone_question')}")
        elif node == "retrieve":
            w(f"  [retrieve] status={update.get('rag_status')}")
            for c in update.get("chunks", []):
                w(f"      d={c['distance']:.3f} | {c['document_title']} > {c['heading_path']}")
        elif node == "prepare":
            w(f"  [prepare] {len(update.get('messages', []))} messages built")
        elif node == "agent":
            calls = getattr(update["messages"][-1], "tool_calls", None) or []
            w("  [agent] wants tools: " + ", ".join(c["name"] for c in calls)
              if calls else "  [agent] final answer ready")
        elif node == "tools":
            for t in update.get("trace", []):
                w(f"  [tools] {t['tool']} ok={t['ok']} args={t['args']}")