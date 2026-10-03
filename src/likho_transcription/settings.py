"""Configuration, read from environment variables and the .env files of the current environment.

LIKHO_ENV (development, staging or production; default development) picks the files. They are
read in this order, each one overriding the one before, and a real environment variable wins
over all of them:

    .env  .env.local  .env.<LIKHO_ENV>  .env.<LIKHO_ENV>.local

The .env.<LIKHO_ENV> files are committed and hold no secrets; the .local files are ignored by
git and hold the secrets of that environment on this machine.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

LIKHO_ENV = os.environ.get("LIKHO_ENV", "development")
ENV_FILES = (".env", ".env.local", f".env.{LIKHO_ENV}", f".env.{LIKHO_ENV}.local")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, extra="ignore")

    likho_env: str = LIKHO_ENV
    log_level: str = "INFO"

    grpc_port: int = 5020
    http_port: int = 4020  # /healthz and /readyz

    # Defaults match the likho-infra local stack.
    nats_url: str = "nats://localhost:4222"
    mongo_url: str = "mongodb://localhost:27017"
    mongo_database: str = "likho_transcription"
    language_grpc_addr: str = "localhost:5030"
    media_grpc_addr: str = "localhost:5010"
    rpc_timeout_seconds: float = 10.0
    download_timeout_seconds: float = 300.0

    # --- speech model ---------------------------------------------------------
    default_model: str = "turbo"
    device: str = "auto"
    compute_type: str = "auto"
    cpu_threads: int = 0

    # --- job queue --------------------------------------------------------------
    # Take jobs from the event bus. Switch off to run only the gRPC API.
    worker_enabled: bool = True
    job_durable: str = "likho-transcription-requested"
    # "all": also take jobs queued while no worker was running. "new": only jobs published from now on.
    job_start: str = "all"
    # A job that is neither acknowledged nor reported "in progress" for this long is given to another worker.
    job_ack_wait_seconds: float = 60.0
    job_heartbeat_seconds: float = 20.0
    # After this many attempts the job is marked failed.
    job_max_deliver: int = 3
    job_retry_delay_seconds: float = 30.0
