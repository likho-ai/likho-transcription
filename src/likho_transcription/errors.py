"""What can go wrong with a job, in the terms the event contract uses."""

# The codes of likho.transcription.failed.v1.
FAILURE_CODES = ("audio_unreadable", "model_unavailable", "cancelled", "internal")


class JobError(Exception):
    """A job cannot be completed.

    code       one of FAILURE_CODES
    message    what happened, in plain words, for the Jobs page
    retryable  True when another attempt may succeed (a service was down); False when it cannot
    """

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        assert code in FAILURE_CODES, code
        self.code = code
        self.message = message
        self.retryable = retryable


class JobCancelled(Exception):  # noqa: N818 - a signal, not an error
    """The job was stopped on request."""
