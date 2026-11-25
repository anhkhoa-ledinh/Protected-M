"""What the UI needs to know about the shared (cloud) album: whether upload
and list/random are configured, and the label to show when they are not."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.content import shared_album


class SharedAlbumService(ABC):
    NOT_CONFIGURED: str

    @abstractmethod
    def can_upload(self) -> bool: ...

    @abstractmethod
    def can_list(self) -> bool: ...


class LocalSharedAlbumService(SharedAlbumService):
    NOT_CONFIGURED = shared_album.NOT_CONFIGURED

    def can_upload(self) -> bool:
        return shared_album.can_upload()

    def can_list(self) -> bool:
        return shared_album.load_config() is not None
