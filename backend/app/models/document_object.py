from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageContent:
    page_number: int
    text: str
    sections: list[str] = field(default_factory=list)


@dataclass
class DocumentObject:
    text: str
    pages: list[PageContent]
    metadata: dict[str, Any]
    document_type: str | None = None
    title: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)
