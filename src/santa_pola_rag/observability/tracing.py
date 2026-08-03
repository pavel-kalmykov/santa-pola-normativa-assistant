from functools import lru_cache

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai.agent import Agent

from santa_pola_rag.config import settings

_configured = False


@lru_cache(maxsize=1)
def setup_tracing() -> trace.Tracer:
    """Configure the global OTel tracer provider (OTLP -> Tempo) and turn on
    pydantic-ai's built-in agent/tool/model-call instrumentation."""
    global _configured
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces"
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    Agent.instrument_all()
    _configured = True

    return trace.get_tracer(settings.otel_service_name)
