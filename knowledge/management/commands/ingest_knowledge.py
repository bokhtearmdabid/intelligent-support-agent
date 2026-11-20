from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from knowledge.ingestion import SUPPORTED_SUFFIXES, deactivate_missing, ingest_source


class Command(BaseCommand):
    help = "Load documents from knowledge_base/ into the vector database."

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true",
                            help="Re-process even if nothing changed.")
        parser.add_argument("--prune", action="store_true",
                            help="Deactivate documents whose files no longer exist.")

    def handle(self, *args, **options):
        base = Path(settings.KNOWLEDGE_DIR)
        if not base.is_dir():
            raise CommandError(f"Knowledge directory not found: {base}")

        names = sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not names:
            raise CommandError("No supported files (.md, .txt, .pdf) found.")

        failed = 0
        for name in names:
            doc, outcome = ingest_source(name, force=options["force"])
            line = f"{outcome:<12} {name}  ({doc.chunk_count} chunks)"
            if outcome == "failed":
                failed += 1
                self.stdout.write(self.style.ERROR(f"{line}\n             {doc.error_message}"))
            elif outcome in ("created", "updated"):
                self.stdout.write(self.style.SUCCESS(line))
            else:
                self.stdout.write(line)

        if options["prune"]:
            count = deactivate_missing(names)
            self.stdout.write(f"Deactivated {count} missing document(s).")

        if failed:
            raise CommandError(f"{failed} document(s) failed.")