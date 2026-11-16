"""LAN PvP networking: protocol, server, discovery, client. TASK_pvp_lan_1116.

Standard library only (``asyncio``, ``socket``, ``json``, ``threading``) —
no new pip packages. Network I/O never runs on the GL thread; see
``src/net/client.py`` for the worker-thread pattern copied from
``src/content/shared_album_worker.py``.
"""

from __future__ import annotations
