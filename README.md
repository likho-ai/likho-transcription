# likho-transcription

The transcription service of [Likho](https://github.com/likho-ai). It turns a call recording into
a transcript with two layers for every line:

| Layer | Field | Example |
| --- | --- | --- |
| 1. As spoken, in the script of the language | `text_script` | `नमस्ते, आपका ऑर्डर कल तक पहुँच जाएगा` |
| 2. Hinglish, the Roman-letter Hindi people type in chat | `text_roman` | `namaste, aapka order kal tak pahunch jayega` |

The repository holds two packages:

| Package | What it is |
| --- | --- |
| `likho_engine` | The speech engine. Audio in, lines out. No network, no database. Also a command line tool. |
| `likho_transcription` | The service around it: takes jobs from the event bus, stores transcripts, answers gRPC calls. |

## How a recording becomes a transcript

1. **Load.** The file is read exactly as it was uploaded; any format FFmpeg reads works. It is
   converted to 16 kHz mono in memory.
2. **Find the speech.** A voice detector marks where people talk. Silences of 3 seconds or more
   are skipped; shorter pauses are kept.
3. **Cut in pauses.** Speech is cut into chunks of at most 12 seconds, always inside a pause, so
   no word is cut in half and no second of speech is dropped.
4. **Detect the language.** The model's five best guesses are kept with the transcript.
   [likho-language](https://github.com/likho-ai/likho-language) decides what to decode as. Urdu,
   for example, is decoded as Hindi so that the text arrives in Devanagari.
5. **Decode.** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) writes each chunk.
   Whisper writes at most 224 tokens per chunk and drops the rest; a chunk that reaches the limit
   is decoded again in two halves. The workspace's glossary is given to the model as names to
   listen for.
6. **Clean.** Repetition loops (`haan haan haan haan ...`) are collapsed.
7. **Write layer 2.** likho-language turns each line into Hinglish with the workspace's spellings.

## Use the engine from the command line

No services needed. The model (about 1.6 GB) is downloaded on first use.

```bash
uv sync
uv run likho-transcribe call.mp3                       # writes transcripts/call.hinglish.txt
uv run likho-transcribe -f txt -f srt -f json call.mp3 # json holds both layers
uv run likho-transcribe --layer script call.mp3        # the Devanagari layer
uv run likho-transcribe --watch                        # transcribe new files in ./recordings as they arrive
uv run likho-transcribe --help
```

`glossary.txt` (names, one per line) and `custom_words.json` (`{"त्रिफला": "Triphala"}`) are used
when they exist in the folder the command runs in.

From Python:

```python
from likho_engine import Engine, EngineSettings, LocalLanguage, transcribe

engine = Engine(EngineSettings())
transcript = transcribe(engine, "call.mp3", LocalLanguage())
for line in transcript.segments:
    print(line.start, line.text_script, "|", line.text_roman)
```

On a 6-core desktop CPU without a GPU, the default model (`turbo`, 8-bit) transcribes about 1.4
times faster than the recording plays.

## Run the service

Needs MongoDB and NATS from the [likho-infra](https://github.com/likho-ai/likho-infra) stack, with
the streams created (`bash scripts/up.sh`).

```bash
uv sync
uv run likho-transcription       # gRPC on 5020, health on 4020, takes jobs from the event bus
```

With Docker, on the stack's network. The volume keeps the downloaded model between runs:

```bash
docker build -t likho-transcription .
docker run --rm --network likho -p 5020:5020 -p 4020:4020 -v likho-models:/models \
  -e MONGO_URL=mongodb://mongo:27017 -e NATS_URL=nats://nats:4222 \
  -e LANGUAGE_GRPC_ADDR=likho-language:5030 -e MEDIA_GRPC_ADDR=likho-media:5010 \
  likho-transcription
```

### Jobs

A job is a `likho.transcription.requested.v1` event on the subject `likho.transcription.requested`:

```json
{
  "specversion": "1.0",
  "id": "evt_01JB7Z5M0A1B2C3D4E5F6G7H8J",
  "source": "likho-api",
  "type": "likho.transcription.requested.v1",
  "time": "2026-10-01T07:00:00Z",
  "data": {
    "job_id": "job_01JB7Z5M4R6T8V0X2Z4B6D8F0H",
    "recording_id": "rec_01JB7Z5KQ2S4U6W8Y0A2C4E6G8",
    "media_id": "med_01JB7Z5KT3V5X7Z9B1D3F5H7K9",
    "workspace_id": "wsp_01JB7Z5K3M9Q2W4X6Y8A0C1E3G",
    "language_policy": "auto",
    "force": false
  }
}
```

The service asks [likho-media](https://github.com/likho-ai/likho-media) for a link to the file as it
was uploaded, transcribes it, and publishes:

| Subject | When |
| --- | --- |
| `likho.live.segment` | Each line, as soon as it is written, with both layers. For the live view. |
| `likho.transcription.completed` | The transcript is stored. Carries its id, version, language and timings. |
| `likho.transcription.failed` | The job cannot be done. Carries a code and a message in plain words. |
| `likho.dead` | The request that failed, with the reason, kept for a person to look at. |

The event schemas are in [likho-contracts](https://github.com/likho-ai/likho-contracts).

What you can rely on:

- **One job at a time per worker.** The model keeps the CPU busy. Start more workers on more
  machines to do more; they share one queue.
- **A job is not lost when a worker dies.** While it works, the worker tells the bus the job is in
  progress. When that stops, the bus gives the job to another worker.
- **A job is not done twice.** A recording that already has a transcript reports that transcript,
  unless the job says `force`. Events carry ids derived from the job, so a repeated event is
  stored once.
- **Failures are sorted.** When a service is down the job is retried, 3 attempts with 30 seconds
  between them. When the audio cannot be read, or the model does not exist, it fails at once.
- **likho-language may be down.** Lines are then written with the built-in rules and the
  transcript is stored with `vocabulary_version` 0. `Retransliterate` applies the workspace's
  spellings later.
- **Stopping is clean.** On SIGTERM the worker finishes the job in hand, then exits.

### gRPC

`likho.transcription.v1.TranscriptionService`, defined in likho-contracts.

| Call | What it does |
| --- | --- |
| `GetTranscript` | One transcript with every line. |
| `ListTranscripts` | Every version for a recording, newest first, without the lines. |
| `Transcribe` | Transcribes now and streams what happens: started, each line, the stored transcript. Closing the stream stops the work. |
| `Retransliterate` | Stores a new version whose Hinglish is rebuilt from the saved script layer. The speech model does not run. |
| `ListEngines` | The models this worker can load. |
| `CancelJob` | Stops a running job after its current line. |

### Storage

MongoDB, database `likho_transcription`, collection `transcripts`. One document per version:

```json
{
  "_id": "trn_01JB7Z6A...",
  "recording_id": "rec_01JB7Z5K...",
  "job_id": "job_01JB7Z5M...",
  "workspace_id": "wsp_01JB7Z5K...",
  "version": 1,
  "model": { "registry_id": "faster-whisper/turbo", "engine": "faster-whisper", "compute": "int8" },
  "language": {
    "detected": "hi",
    "probability": 0.91,
    "candidates": [{ "language": "hi", "probability": 0.91 }, { "language": "ur", "probability": 0.07 }],
    "decoded_as": "hi",
    "policy": "auto"
  },
  "script": "devanagari",
  "transliterated": true,
  "vocabulary_version": 3,
  "segments": [
    { "index": 0, "start": 0.4, "end": 3.1, "text_script": "नमस्ते", "text_roman": "namaste" }
  ],
  "stats": { "audio_seconds": 61.2, "elapsed_seconds": 44.0, "realtime_factor": 1.39, "chunks": 6, "silence_skipped_seconds": 4.5 },
  "created_at": "2026-10-01T07:01:02Z"
}
```

A version is never changed. A correction or a re-transliteration is a new version.

## Configuration

Environment variables, or a `.env` file (see `.env.example`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `GRPC_PORT` | `5020` | gRPC, including the standard health service |
| `HTTP_PORT` | `4020` | `GET /healthz` (alive), `GET /readyz` (MongoDB and NATS answer) |
| `MONGO_URL` | `mongodb://localhost:27017` | Transcript store |
| `MONGO_DATABASE` | `likho_transcription` | |
| `NATS_URL` | `nats://localhost:4222` | Event bus |
| `LANGUAGE_GRPC_ADDR` | `localhost:5030` | likho-language |
| `MEDIA_GRPC_ADDR` | `localhost:5010` | likho-media |
| `DEFAULT_MODEL` | `turbo` | `turbo`, `large-v3`, `medium`, `small`, `base` |
| `DEVICE` | `auto` | `auto` uses a CUDA GPU when there is one, otherwise the CPU |
| `COMPUTE_TYPE` | `auto` | `auto` is `float16` on a GPU and `int8` on a CPU |
| `CPU_THREADS` | `0` | Threads for decoding on a CPU. 0 = the library's default (4) |
| `WORKER_ENABLED` | `true` | `false` runs only the gRPC API |
| `JOB_START` | `all` | `all` also takes jobs queued while no worker ran; `new` takes only later ones |
| `JOB_MAX_DELIVER` | `3` | Attempts before a job is marked failed |
| `JOB_RETRY_DELAY_SECONDS` | `30` | Wait between attempts |
| `JOB_ACK_WAIT_SECONDS` | `60` | A silent worker loses its job after this long |
| `JOB_HEARTBEAT_SECONDS` | `20` | How often a working worker reports "in progress" |
| `LOG_LEVEL` | `INFO` | Logs are JSON, one object per line |

## Develop

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest
```

The tests use stand-ins for the speech model, so they need no download and run in seconds. The
service tests run the real service against MongoDB and NATS from likho-infra; without the stack
they are skipped, and with `LIKHO_REQUIRE_STACK=1` (set in CI) they fail instead.

Recordings and transcripts are data. The folders `recordings/` and `transcripts/` are ignored by
git, and no real call audio or text belongs in this repository.
