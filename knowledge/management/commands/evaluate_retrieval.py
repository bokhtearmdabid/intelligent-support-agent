import statistics
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from knowledge.embeddings import EmbeddingError
from knowledge.evaluation import evaluate, load_cases, suggest_threshold

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "eval" / "retrieval_eval.json"


def _short(text, n=55):
    return text if len(text) <= n else text[: n - 1] + "…"


class Command(BaseCommand):
    help = "Measure retrieval quality and suggest RAG_MAX_DISTANCE."

    def add_arguments(self, parser):
        parser.add_argument("--file", default=str(DEFAULT_FILE))
        parser.add_argument("--k", type=int, default=settings.RAG_TOP_K)

    def handle(self, *args, **options):
        cases = load_cases(options["file"])
        try:
            results = evaluate(cases, options["k"])
        except EmbeddingError:
            raise CommandError("Embedding API failed. Check GOOGLE_API_KEY and the log above.")

        answerable = [r for r in results if r.kind == "answerable"]
        out_scope = [r for r in results if r.kind == "out_of_scope"]

        self.stdout.write("\n== Answerable questions ==")
        for r in answerable:
            if r.rank:
                self.stdout.write(f"  OK   rank={r.rank} d={r.hit_distance:.3f} | {_short(r.question)}")
            else:
                self.stdout.write(self.style.ERROR(
                    f"  MISS top1 d={r.top_distance:.3f} [{r.top_label}] | {_short(r.question)}"))

        self.stdout.write("\n== Out-of-scope questions (no answer in documents) ==")
        for r in out_scope:
            self.stdout.write(f"  top1 d={r.top_distance:.3f} [{r.top_label}] | {_short(r.question)}")

        n = len(answerable)
        hit1 = sum(1 for r in answerable if r.rank == 1)
        hitk = sum(1 for r in answerable if r.rank)
        mrr = sum(1 / r.rank for r in answerable if r.rank) / n if n else 0.0
        self.stdout.write(f"\nHit@1 = {hit1}/{n}   Hit@{options['k']} = {hitk}/{n}   MRR = {mrr:.2f}")

        hits = [r.hit_distance for r in answerable if r.hit_distance is not None]
        outs = [r.top_distance for r in out_scope if r.top_distance is not None]
        if hits:
            self.stdout.write(f"Answerable hit distance : min={min(hits):.3f} median={statistics.median(hits):.3f} max={max(hits):.3f}")
        if outs:
            self.stdout.write(f"Out-of-scope top1 dist. : min={min(outs):.3f} median={statistics.median(outs):.3f} max={max(outs):.3f}")

        suggested, separable = suggest_threshold(hits, outs)
        if suggested is None:
            self.stdout.write(self.style.WARNING("No correct hits: fix retrieval before choosing a threshold."))
        elif separable:
            self.stdout.write(self.style.SUCCESS(f"\nSeparable. Suggested RAG_MAX_DISTANCE = {suggested}"))
        else:
            self.stdout.write(self.style.WARNING(
                f"\nOverlap: some out-of-scope questions are as close as real answers.\n"
                f"A distance threshold alone cannot separate them. Suggested (recall-first): {suggested}\n"
                f"The prompt's 'say you don't know' rule is the second line of defence."))
        self.stdout.write(f"Current setting: RAG_MAX_DISTANCE = {settings.RAG_MAX_DISTANCE}")