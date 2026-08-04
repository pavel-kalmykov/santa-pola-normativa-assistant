from functools import lru_cache

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai.agent import Agent

from santa_pola_rag.config import settings

_configured = False


def _parse_otlp_headers(raw: str | None) -> dict[str, str] | None:
    """Parse the standard OTEL_EXPORTER_OTLP_HEADERS format ("k1=v1,k2=v2"),
    e.g. Grafana Cloud's "Authorization=Basic <base64>" for its Tempo OTLP
    endpoint; unset for a local, unauthenticated Tempo."""
    if not raw:
        return None
    return dict(pair.split("=", 1) for pair in raw.split(",") if "=" in pair)


@lru_cache(maxsize=1)
def setup_tracing() -> trace.Tracer:
    """Configure the global OTel tracer provider (OTLP -> Tempo) and turn on
    pydantic-ai's built-in agent/tool/model-call instrumentation."""
    global _configured
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces",
        headers=_parse_otlp_headers(settings.otel_exporter_otlp_headers),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    Agent.instrument_all()
    _configured = True

    return trace.get_tracer(settings.otel_service_name)
