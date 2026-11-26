from django.contrib.auth.models import AnonymousUser
from django.core.management.base import BaseCommand

from support.graph import build_graph


class Command(BaseCommand):
    help = "Print the agent graph as Mermaid text (paste into README or mermaid.live)."

    def handle(self, *args, **options):
        self.stdout.write(build_graph(AnonymousUser()).get_graph().draw_mermaid())