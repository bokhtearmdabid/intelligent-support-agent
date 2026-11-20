from django.core.management.base import BaseCommand, CommandError

from support.llm import LLMUnavailable, generate_reply


class Command(BaseCommand):
    help = "Send one message to the configured LLM and print the reply."

    def add_arguments(self, parser):
        parser.add_argument("message")

    def handle(self, *args, **options):
        try:
            result = generate_reply([], options["message"])
        except LLMUnavailable:
            raise CommandError(
                "LLM call failed. Check the log above, your API key and LLM_MODEL."
            )
        self.stdout.write(result.text)
        self.stdout.write(self.style.SUCCESS(
            f"\n[model={result.model_name} prompt={result.prompt_version} "
            f"in={result.input_tokens} out={result.output_tokens}]"
        ))