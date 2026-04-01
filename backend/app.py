"""FastAPI application — serves the REST API and static HTML frontend."""

from __future__ import annotations

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from backend.models import NoteCreate, NoteUpdate, NoteOut, AIResult
from backend.store import note_store
from backend.ai_service import AIAction, run_ai_action, is_ai_enabled

app = FastAPI(title="Smart Notes", version="0.1.0")

# ── OpenTelemetry instrumentation ────────────────────────────────────────────

_otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "https://production-otlp-2fa05823.app.embr.azure"
_prom_endpoint = "https://production-prometheus-embr-1a780423.app.embr.azure/api/v1/otlp/v1/metrics"
_notes_counter = None

if _otel_endpoint:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    _resource = Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "simple-python-notes-app"),
    })

    # Traces
    _tp = TracerProvider(resource=_resource)
    _tp.add_span_processor(SimpleSpanProcessor(
        OTLPSpanExporter(endpoint=_otel_endpoint + "/v1/traces")
    ))
    trace.set_tracer_provider(_tp)
    FastAPIInstrumentor.instrument_app(app)

    # Logs
    _lp = LoggerProvider(resource=_resource)
    _lp.add_log_record_processor(SimpleLogRecordProcessor(
        OTLPLogExporter(endpoint=_otel_endpoint + "/v1/logs")
    ))
    set_logger_provider(_lp)
    logging.getLogger().addHandler(
        LoggingHandler(level=logging.INFO, logger_provider=_lp)
    )
    logging.getLogger().setLevel(logging.INFO)

    # Metrics — send to Prometheus (or OTLP endpoint if no separate metrics endpoint)
    _metrics_url = _prom_endpoint or (_otel_endpoint + "/v1/metrics")
    _reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=_metrics_url),
        export_interval_millis=15000,
    )
    _mp = MeterProvider(resource=_resource, metric_readers=[_reader])
    metrics.set_meter_provider(_mp)

    _meter = _mp.get_meter("smart_notes")
    _notes_counter = _meter.create_counter(
        "notes.created",
        description="Number of notes created",
    )

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
logger = logging.getLogger("smart_notes")

# ── API routes ───────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "ai_enabled": is_ai_enabled()}


@app.post("/api/notes", response_model=NoteOut, status_code=201)
async def create_note(data: NoteCreate):
    note = note_store.create(data).to_out()
    logger.info("Note created: %s", note.id)
    if _notes_counter:
        _notes_counter.add(1)
    return note


@app.get("/api/notes", response_model=list[NoteOut])
async def list_notes():
    return [n.to_out() for n in note_store.list_all()]


@app.get("/api/notes/{note_id}", response_model=NoteOut)
async def get_note(note_id: str):
    rec = note_store.get(note_id)
    if rec is None:
        raise HTTPException(404, "Note not found")
    return rec.to_out()


@app.put("/api/notes/{note_id}", response_model=NoteOut)
async def update_note(note_id: str, data: NoteUpdate):
    rec = note_store.update(note_id, data)
    if rec is None:
        raise HTTPException(404, "Note not found")
    return rec.to_out()


@app.delete("/api/notes/{note_id}", status_code=204)
async def delete_note(note_id: str):
    if not note_store.delete(note_id):
        raise HTTPException(404, "Note not found")
    logger.info("Note deleted: %s", note_id)


@app.post("/api/notes/{note_id}/ai/{action}", response_model=AIResult)
async def ai_action(note_id: str, action: AIAction):
    rec = note_store.get(note_id)
    if rec is None:
        raise HTTPException(404, "Note not found")
    result = await run_ai_action(action, rec.content)
    return AIResult(note_id=note_id, action=action.value, result=result)


# ── Serve static HTML frontend ───────────────────────────────────────────────

@app.get("/favicon.svg")
async def serve_favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")
