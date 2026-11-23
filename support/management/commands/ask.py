from django.core.management.base import BaseCommand, CommandError

from accounts.models import Customer
from support.llm import LLMUnavailable, generate_reply
from support.rag import retrieve_context
from support.tools import build_tools


class Command(BaseCommand):
    help = "Ask the assistant one question (RAG + tools) and show what happened."

    def add_arguments(self, parser):
        parser.add_argument("message")
        parser.add_argument("--as", dest="as_email", default=None,
                            help="Act as this customer's email (dev only). Without it: no tools.")

    def handle(self, *args, **options):
        message = options["message"]
        tools = None
        if options["as_email"]:
            try:
                user = Customer.objects.get(email=options["as_email"])
            except Customer.DoesNotExist:
                raise CommandError(f"No customer with email {options['as_email']}")
            tools = build_tools(user)
            self.stdout.write(f"acting as: {user.email}")
        else:
            self.stdout.write("acting as: anonymous (no tools)")

        context = retrieve_context([], message)
        self.stdout.write(f"retrieval status: {context.status}")
        for chunk in context.chunks:
            self.stdout.write(
                f"  d={chunk.distance:.3f} | {chunk.document_title} > {chunk.heading_path}"
            )

        try:
            result = generate_reply([], message, context=context, tools=tools)
        except LLMUnavailable:
            raise CommandError("LLM call failed. Check the log above.")

        for t in result.tool_trace:
            self.stdout.write(f"  tool={t['tool']} ok={t['ok']} args={t['args']}")

        self.stdout.write(f"\n{result.text}\n")
        self.stdout.write(self.style.SUCCESS(
            f"[prompt={result.prompt_version} in={result.input_tokens} "
            f"out={result.output_tokens} tool_calls={len(result.tool_trace)}]"
        ))