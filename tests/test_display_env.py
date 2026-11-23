import unittest

from src.display_env import prefer_xwayland

WAYLAND = {"WAYLAND_DISPLAY": "wayland-1", "DISPLAY": ":0"}


class PreferXwaylandTests(unittest.TestCase):
    def test_wayland_with_xwayland_selects_x11(self) -> None:
        env = dict(WAYLAND)
        self.assertTrue(prefer_xwayland(env, "linux"))
        self.assertEqual(env["PYGLFW_LIBRARY_VARIANT"], "x11")
        self.assertEqual(env["PYOPENGL_PLATFORM"], "glx")

    def test_windows_is_untouched(self) -> None:
        env = dict(WAYLAND)
        self.assertFalse(prefer_xwayland(env, "win32"))
        self.assertEqual(env, WAYLAND)

    def test_wayland_without_xwayland_is_untouched(self) -> None:
        env = {"WAYLAND_DISPLAY": "wayland-1"}
        self.assertFalse(prefer_xwayland(env, "linux"))
        self.assertEqual(env, {"WAYLAND_DISPLAY": "wayland-1"})

    def test_x11_session_is_untouched(self) -> None:
        env = {"DISPLAY": ":0"}
        self.assertFalse(prefer_xwayland(env, "linux"))
        self.assertEqual(env, {"DISPLAY": ":0"})

    def test_user_choice_is_kept(self) -> None:
        env = dict(WAYLAND, PYGLFW_LIBRARY_VARIANT="wayland", PYOPENGL_PLATFORM="egl")
        self.assertFalse(prefer_xwayland(env, "linux"))
        self.assertEqual(env["PYGLFW_LIBRARY_VARIANT"], "wayland")
        self.assertEqual(env["PYOPENGL_PLATFORM"], "egl")


if __name__ == "__main__":
    unittest.main()
