"""The exception a task raises at its own cancel points.

In ``core`` so pipeline code (``rag``, ``core``, subprocess children) can
raise it without importing the API layer; ``api.tasks`` re-exports it.
"""

from __future__ import annotations


class TaskCancelled(Exception):
    """Raised at one of a task's own cancel points once the user cancelled it.

    *summary* says what the task finished before it stopped ("3 downloaded");
    ``TaskManager`` shows it after "Cancelled" as the final message. Not a
    failure: nothing is logged as an error.
    """

    def __init__(self, summary: str = "") -> None:
        super().__init__(summary or "Cancelled")
        self.summary = summary
