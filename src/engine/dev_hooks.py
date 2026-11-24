"""``DevHooks``: the small public face of ``HcpdApplication`` that the
developer tools (``src/devtools``) are allowed to use.

Devtools used to reach into the app's private attributes (``app._album``,
``app._window``, ...). They now go through this facade, so renaming an app
internal means changing one line here, not 200 across the devtools.
``tests/test_architecture.py`` forbids ``app._x`` access under ``src/devtools``.
"""

from __future__ import annotations

#: Attributes of the app the devtools read; the first three are also assigned.
_READ_ONLY = ("album", "window", "root", "album_options", "renderer", "bugs",
              "last_scene", "content")
_READ_WRITE = ("game", "game_planes", "game_layers_revision")
#: App methods the devtools call.
_METHODS = ("tick_scope", "album_draw_guarded", "try_load_media", "save_album_shot",
            "v3_field_focused", "start_album_match", "pvp_action", "album_shoot",
            "album_action")


class DevHooks:
    def __init__(self, app) -> None:
        self._app = app


def _reader(name):
    return property(lambda self: getattr(self._app, "_" + name))


def _accessor(name):
    return property(lambda self: getattr(self._app, "_" + name),
                    lambda self, value: setattr(self._app, "_" + name, value))


def _method(name):
    def call(self, *args, **kwargs):
        return getattr(self._app, "_" + name)(*args, **kwargs)
    call.__name__ = name
    return call


for _name in _READ_ONLY:
    setattr(DevHooks, _name, _reader(_name))
for _name in _READ_WRITE:
    setattr(DevHooks, _name, _accessor(_name))
for _name in _METHODS:
    setattr(DevHooks, _name, _method(_name))
