"""Streamlit progress adapter for the deep trading pipeline."""

from __future__ import annotations

from typing import Any


class StreamlitProgressPrinter:
    """Render pipeline status updates into one Streamlit placeholder."""

    def __init__(self, container: Any) -> None:
        self.container = container
        self.items: dict[str, tuple[str, bool]] = {}
        self._placeholder = container.empty()

    def update_item(self, item_id: str, content: str, is_done: bool = False) -> None:
        self.items[item_id] = (content, is_done)
        self.flush()

    def mark_item_done(self, item_id: str, content: str | None = None) -> None:
        if item_id not in self.items:
            return
        previous_content, _ = self.items[item_id]
        self.items[item_id] = (content or previous_content, True)
        self.flush()

    def fail_item(self, item_id: str, content: str) -> None:
        self.items[item_id] = (content, True)
        self.flush()

    def flush(self) -> None:
        lines = []
        for content, is_done in self.items.values():
            prefix = "- [x] " if is_done else "- [ ] "
            lines.append(prefix + content)
        self._placeholder.markdown("\n".join(lines) if lines else "_Waiting..._")
