from django.core.management.base import BaseCommand, CommandError

from support.llm import LLMUnavailable, generate_reply
from support.rag import retrieve_context


class Command(BaseCommand):
    help = "Ask the assistant one question (with RAG) and show what was retrieved."

    def add_arguments(self, parser):
        parser.add_argument("message")

    def handle(self, *args, **options):
        message = options["message"]
        context = retrieve_context([], message)

        self.stdout.write(f"\nretrieval status: {context.status}")
        for chunk in context.chunks:
            self.stdout.write(
                f"  d={chunk.distance:.3f} | {chunk.document_title} > {chunk.heading_path}"
            )

        try:
            result = generate_reply([], message, context=context)
        except LLMUnavailable:
            raise CommandError("LLM call failed. Check the log above.")

        self.stdout.write(f"\n{result.text}\n")
        self.stdout.write(self.style.SUCCESS(
            f"[prompt={result.prompt_version} in={result.input_tokens} out={result.output_tokens}]"
        ))