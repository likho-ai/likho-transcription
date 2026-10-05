"""What the worker tells about itself (OpenTelemetry): always as Prometheus text at GET /metrics,
and pushed over OTLP/HTTP as well when OTEL_EXPORTER_OTLP_ENDPOINT is set."""

from __future__ import annotations

import atexit
import logging

from opentelemetry import metrics as otel
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

log = logging.getLogger(__name__)

SERVICE = "likho-transcription"

_shared: Metrics | None = None


def shared(version: str, otlp_endpoint: str = "") -> Metrics:
    """The process's one set of instruments (Prometheus has one registry per process)."""
    global _shared
    if _shared is None:
        _shared = Metrics(version, otlp_endpoint)
        otel.set_meter_provider(_shared._provider)
        atexit.register(_shared._provider.shutdown)
    return _shared


class Metrics:
    """The instruments the worker records into."""

    def __init__(self, version: str, otlp_endpoint: str = "") -> None:
        readers: list[MetricReader] = [PrometheusMetricReader()]
        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

            url = otlp_endpoint.rstrip("/") + "/v1/metrics"
            readers.append(
                PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=url), export_interval_millis=15_000)
            )
            log.info("metrics go to %s every 15 s, and are at /metrics", url)
        self._provider = MeterProvider(
            resource=Resource.create({SERVICE_NAME: SERVICE, SERVICE_VERSION: version}),
            metric_readers=readers,
            views=[
                View(
                    instrument_name="likho_job_realtime_factor",
                    aggregation=ExplicitBucketHistogramAggregation((0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)),
                ),
                View(
                    instrument_name="likho_job_seconds",
                    aggregation=ExplicitBucketHistogramAggregation((5, 15, 30, 60, 120, 300, 600, 1200, 3600)),
                ),
                View(
                    instrument_name="likho_model_load_seconds",
                    aggregation=ExplicitBucketHistogramAggregation((1, 2, 5, 10, 20, 30, 60, 120)),
                ),
            ],
        )
        meter = self._provider.get_meter(SERVICE, version)
        #: Jobs that ended, by outcome: done, failed, cancelled, retry.
        self.jobs_finished = meter.create_counter("likho_jobs_finished", description="Jobs that ended, by outcome")
        #: Audio seconds transcribed per wall-clock second (1 = real time).
        self.realtime_factor = meter.create_histogram(
            "likho_job_realtime_factor", description="Audio seconds transcribed per second of wall-clock time"
        )
        #: How long a job took, in seconds.
        self.job_seconds = meter.create_histogram("likho_job_seconds", description="How long a job took")
        #: Lines published while a job ran.
        self.segments = meter.create_counter("likho_segments_published", description="Lines published while jobs ran")
        #: How long loading a model took, by model size.
        self.model_load_seconds = meter.create_histogram(
            "likho_model_load_seconds", description="How long loading a speech model took"
        )
        #: Jobs running right now (0 or 1 per worker).
        self.jobs_running = meter.create_up_down_counter("likho_jobs_running", description="Jobs running right now")

    def scrape(self) -> tuple[str, bytes]:
        """The Prometheus text and its content type."""
        return CONTENT_TYPE_LATEST, generate_latest(REGISTRY)

    def flush(self) -> None:
        """Sends what is pending to the OTLP endpoint, if any (at a stop)."""
        self._provider.force_flush()
