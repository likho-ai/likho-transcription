"""Transcripts in MongoDB: one document per version, with both text layers of every line."""

from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.errors import DuplicateKeyError

from likho_transcription.ids import new_id

Document = dict[str, Any]


class TranscriptStore:
    def __init__(self, url: str, database: str) -> None:
        self._client: AsyncMongoClient[Document] = AsyncMongoClient(url, tz_aware=True, serverSelectionTimeoutMS=5000)
        self._transcripts = self._client[database]["transcripts"]

    async def prepare(self) -> None:
        """Create the indexes. Fails when the database is not reachable."""
        await self._transcripts.create_index([("recording_id", ASCENDING), ("version", DESCENDING)], unique=True)
        # One transcript per job. Versions made by re-transliteration have no job.
        await self._transcripts.create_index(
            "job_id", unique=True, partialFilterExpression={"job_id": {"$gt": ""}}, name="job_id_unique"
        )

    async def ping(self) -> bool:
        await self._client.admin.command("ping")
        return True

    async def close(self) -> None:
        await self._client.close()

    async def insert(self, document: Document) -> Document:
        """Store a new version for its recording and return it with its id, version and creation time."""
        document = {**document, "_id": new_id("trn"), "created_at": datetime.now(UTC)}
        for _ in range(5):  # two writers for one recording: the loser takes the next number
            latest = await self._transcripts.find_one(
                {"recording_id": document["recording_id"]}, sort=[("version", DESCENDING)], projection={"version": 1}
            )
            document["version"] = (latest["version"] if latest else 0) + 1
            try:
                await self._transcripts.insert_one(document)
                return document
            except DuplicateKeyError as error:
                if "job_id_unique" in str(error):
                    raise
        raise RuntimeError(f"could not assign a version for recording {document['recording_id']}")

    async def get(self, transcript_id: str) -> Document | None:
        return await self._transcripts.find_one({"_id": transcript_id})

    async def find_by_job(self, job_id: str) -> Document | None:
        return await self._transcripts.find_one({"job_id": job_id}) if job_id else None

    async def latest_for_recording(self, recording_id: str) -> Document | None:
        return await self._transcripts.find_one({"recording_id": recording_id}, sort=[("version", DESCENDING)])

    async def list_for_recording(self, recording_id: str) -> list[Document]:
        """Every version, newest first, without the lines."""
        cursor = self._transcripts.find({"recording_id": recording_id}, projection={"segments": 0}).sort(
            "version", DESCENDING
        )
        return await cursor.to_list()

    async def delete_recording(self, recording_id: str) -> int:
        result = await self._transcripts.delete_many({"recording_id": recording_id})
        return result.deleted_count
