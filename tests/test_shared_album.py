"""
TASK_shared_album_1111 Phase 2 + hardening (11/11): ``shared_album.py``,
mạng giả.

Không có ``assets/config/shared_album.json`` thật (chưa ai điền cloud_name)
nên không bài nào ở đây chạm mạng thật — mọi ``urllib.request.urlopen`` đều
bị thế bằng giả, đúng luật "Tests with the network mocked".

Phần "fail on the current code" (public_id ``a/b``/``../../x``/khoảng
trắng, thân 25 MB, response thiếu "version") chấm đúng bốn điều khoản hardening:
``download()`` không bao giờ dựng đường dẫn từ ``public_id`` thẳng, URL qua
``urllib.parse.quote``, tải có trần, và không hàm nào ném ngoại lệ.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

from src.content import shared_album as sa


def _hashed_name(public_id: str, fmt: str) -> str:
    return f"{hashlib.sha1(public_id.encode('utf-8')).hexdigest()}.{fmt}"


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "shared_album.json"
    path.write_text(json.dumps({"cloud_name": "demo-cloud",
                                "upload_preset": "demo-preset",
                                "tag": "shared_album"}), encoding="utf-8")
    monkeypatch.setattr(sa, "CONFIG_PATH", path)
    return path


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (40, 30), "blue")
    exif = image.getexif()
    exif[271] = "Acme"      # Make — EXIF phai mat sau khi re-encode
    exif[34853] = {1: "N", 2: (10, 0, 0)}  # GPSInfo — chinh ly do phai xoa
    image.save(path, "JPEG", exif=exif)
    return path


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._pos = 0

    def read(self, n: int | None = None) -> bytes:
        if n is None:
            chunk, self._pos = self._payload[self._pos:], len(self._payload)
            return chunk
        chunk = self._payload[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


# ---- load_config / can_upload --------------------------------------------

def test_missing_file_disables_feature(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "CONFIG_PATH", tmp_path / "nope.json")
    assert sa.load_config() is None


@pytest.mark.parametrize("missing", ["cloud_name", "tag"])
def test_empty_cloud_name_or_tag_disables_everything(tmp_path, monkeypatch, missing):
    fields = {"cloud_name": "demo", "upload_preset": "preset", "tag": "shared"}
    fields[missing] = ""
    path = tmp_path / "shared_album.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    monkeypatch.setattr(sa, "CONFIG_PATH", path)
    assert sa.load_config() is None
    assert sa.list_remote() is None


def test_empty_upload_preset_disables_only_upload(tmp_path, monkeypatch, photo):
    path = tmp_path / "shared_album.json"
    path.write_text(json.dumps({"cloud_name": "demo-cloud", "upload_preset": "",
                                "tag": "shared_album"}), encoding="utf-8")
    monkeypatch.setattr(sa, "CONFIG_PATH", path)

    config = sa.load_config()
    assert config is not None            # cloud_name + tag la du cho list/download
    assert not sa.can_upload(config)
    assert sa.upload(photo) == sa.NOT_CONFIGURED

    seen = {}

    def fake_urlopen(url, timeout=None, context=None):
        seen["url"] = url
        return _FakeResponse(json.dumps({"resources": []}).encode("utf-8"))

    monkeypatch.setattr(sa.urllib.request, "urlopen", fake_urlopen)
    assert sa.list_remote() == []        # that su di ra mang, khong bi chan som
    assert "url" in seen


def test_valid_config_loads(config):
    assert sa.load_config() == {"cloud_name": "demo-cloud",
                                "upload_preset": "demo-preset",
                                "tag": "shared_album"}


def test_upload_without_config_is_not_configured(tmp_path, monkeypatch, photo):
    monkeypatch.setattr(sa, "CONFIG_PATH", tmp_path / "nope.json")
    assert sa.upload(photo) == sa.NOT_CONFIGURED


# ---- multipart body: no key, has the two expected fields ----------------

def test_multipart_has_preset_and_file_and_no_key_like_field():
    body, content_type = sa._multipart({"upload_preset": "demo-preset"},
                                       "photo.jpg", b"\xff\xd8\xff-fake-jpeg")
    assert content_type.startswith("multipart/form-data; boundary=")
    text = body.decode("latin-1")
    assert 'name="upload_preset"' in text and "demo-preset" in text
    assert 'name="file"; filename="photo.jpg"' in text
    assert b"\xff\xd8\xff-fake-jpeg" in body
    for forbidden in ("api_key", "signature", "secret", "api_secret"):
        assert forbidden not in text.lower()


# ---- re-encode: EXIF gone -------------------------------------------------

def test_reencode_strips_exif(photo):
    content = sa._reencode(photo)
    assert content is not None
    result = Image.open(io.BytesIO(content))
    assert result.format == "JPEG"
    assert len(result.getexif()) == 0


def test_reencode_rejects_a_non_image(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a photo", encoding="utf-8")
    assert sa._reencode(path) is None


def test_upload_rejects_a_non_image(config, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a photo", encoding="utf-8")
    assert sa.upload(path) == "Not a photo"


def test_upload_rejects_oversized_result(config, photo, monkeypatch):
    monkeypatch.setattr(sa, "MAX_BYTES", 10)   # bat ky anh that nao cung > 10 byte
    assert sa.upload(photo) == "Photo too large after re-encoding"


# ---- network: error result, never an exception ---------------------------

def test_upload_timeout_is_an_error_result_not_an_exception(config, photo,
                                                            monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(sa.urllib.request, "urlopen", boom)
    result = sa.upload(photo)          # khong duoc nem ngoai le
    assert result == "No connection"


def test_upload_http_500_is_an_error_result_not_an_exception(config, photo,
                                                              monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError("https://api.cloudinary.com", 500,
                                     "Internal Server Error", {}, None)

    monkeypatch.setattr(sa.urllib.request, "urlopen", boom)
    result = sa.upload(photo)
    assert "500" in result and "failed" in result.lower()


@pytest.mark.parametrize("status", [401, 404])
def test_blocked_list_has_distinct_error(config, monkeypatch, tmp_path, status):
    def boom(*a, **k):
        raise urllib.error.HTTPError('https://example.invalid', status, 'Blocked',
            {'x-cld-error': 'Resources of type list are restricted in this account'}, None)
    monkeypatch.setattr(sa.urllib.request, 'urlopen', boom)
    expected = 'Album list is blocked in the Cloudinary account'
    assert sa.list_remote() == expected
    assert sa.pick_random_and_download(tmp_path) == expected


@pytest.mark.parametrize("body,header,message", [
    (b'{"error":{"message":"Upload preset not found"}}', 'fallback', 'Upload preset not found'),
    (b'not json', 'Upload preset not found', 'Upload preset not found'),
    (b'{"error":null}', 'Account blocked', 'Account blocked'),
    (b'{"error":{"message":"' + b'x' * 100 + b'"}}', '', 'x' * 100),
])
def test_upload_http_error_uses_cloudinary_message(config, photo, monkeypatch,
                                                  body, header, message):
    def boom(*a, **k):
        raise urllib.error.HTTPError('https://example.invalid', 400, 'Bad request',
                                     {'x-cld-error': header}, io.BytesIO(body))
    monkeypatch.setattr(sa.urllib.request, 'urlopen', boom)
    assert sa.upload(photo) == ('Upload failed: ' + message)[:60]


def test_upload_success_builds_url_from_response(config, photo, monkeypatch):
    payload = json.dumps({"public_id": "abc123", "version": "17", "format": "jpg"})

    def fake_urlopen(request, timeout=None, context=None):
        assert "upload_preset" in request.data.decode("latin-1")
        return _FakeResponse(payload.encode("utf-8"))

    monkeypatch.setattr(sa.urllib.request, "urlopen", fake_urlopen)
    url = sa.upload(photo)
    assert url == ("https://res.cloudinary.com/demo-cloud/image/upload/"
                   "v17/abc123.jpg")


def test_image_url_keeps_folder_slashes_in_public_id():
    """Cloudinary folders are real "/" path segments in the delivery URL —
    quote(safe="/") must NOT turn "folder/name" into "folder%2Fname". The
    file written to disk by download() stays hash-based regardless (see
    test_download_never_builds_a_path_from_public_id)."""
    item = {"public_id": "folder/name", "version": "1", "format": "jpg"}
    url = sa._image_url("demo-cloud", item)
    assert url == ("https://res.cloudinary.com/demo-cloud/image/upload/"
                   "v1/folder/name.jpg")


def test_upload_response_missing_version_is_an_error_result(config, photo,
                                                             monkeypatch):
    payload = json.dumps({"public_id": "abc123", "format": "jpg"})  # thieu version

    monkeypatch.setattr(sa.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(payload.encode("utf-8")))
    result = sa.upload(photo)          # khong duoc nem KeyError
    assert result == "Upload response was malformed"


# ---- list_remote / download: same "error result" contract ----------------

def test_list_remote_without_config_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "CONFIG_PATH", tmp_path / "nope.json")
    assert sa.list_remote() is None


def test_list_remote_network_failure_returns_none(config, monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(sa.urllib.request, "urlopen", boom)
    assert sa.list_remote() is None


def test_download_skips_an_existing_file(config, tmp_path):
    dest_dir = tmp_path / "cache"
    dest_dir.mkdir()
    existing = dest_dir / _hashed_name("abc123", "jpg")
    existing.write_bytes(b"already here")
    item = {"public_id": "abc123", "version": "1", "format": "jpg"}
    assert sa.download(item, dest_dir) == existing
    assert existing.read_bytes() == b"already here"   # khong tai lai de


# ---- hardening 11/11: remote data (public_id/format/size) is untrusted ---

@pytest.mark.parametrize("public_id", ["a/b", "../../x", "evil name"])
def test_download_never_builds_a_path_from_public_id(config, tmp_path,
                                                      monkeypatch, public_id):
    """``a/b``, ``../../x``, khoảng trắng: KHÔNG được tạo thư mục con hay
    thoát khỏi ``dest_dir`` — tên file luôn là sha1(public_id)."""
    dest_dir = tmp_path / "cache"
    dest_dir.mkdir()
    item = {"public_id": public_id, "version": "1", "format": "jpg"}

    monkeypatch.setattr(sa.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(b"\xff\xd8\xff-fake"))
    result = sa.download(item, dest_dir)

    assert result is not None
    assert result.name == _hashed_name(public_id, "jpg")
    assert result.parent.resolve() == dest_dir.resolve()
    assert result.resolve().is_relative_to(dest_dir.resolve())
    # Tuyet doi khong tao thu muc con nao (vd "a" cho public_id "a/b"):
    assert [p.name for p in dest_dir.iterdir()] == [result.name]


def test_download_quotes_public_id_with_a_space(config, tmp_path, monkeypatch):
    dest_dir = tmp_path / "cache"
    dest_dir.mkdir()
    item = {"public_id": "my photo", "version": "1", "format": "jpg"}
    seen = {}

    def fake_urlopen(url, timeout=None, context=None):
        seen["url"] = url
        return _FakeResponse(b"\xff\xd8\xff-ok")

    monkeypatch.setattr(sa.urllib.request, "urlopen", fake_urlopen)
    result = sa.download(item, dest_dir)
    assert result is not None
    assert " " not in seen["url"]
    assert "my%20photo" in seen["url"]


def test_download_aborts_past_20mb_and_writes_nothing(config, tmp_path, monkeypatch):
    dest_dir = tmp_path / "cache"
    dest_dir.mkdir()
    item = {"public_id": "big", "version": "1", "format": "jpg"}
    oversized = b"x" * (25 * 1024 * 1024)

    monkeypatch.setattr(sa.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(oversized))
    result = sa.download(item, dest_dir)
    assert result is None
    assert list(dest_dir.iterdir()) == []   # khong co file nua-vo nao sot lai


def test_download_rejects_a_format_outside_the_allowlist(config, tmp_path):
    dest_dir = tmp_path / "cache"
    dest_dir.mkdir()
    item = {"public_id": "abc", "version": "1", "format": "exe"}
    assert sa.download(item, dest_dir) is None


def test_download_item_missing_version_is_an_error_result_not_an_exception(
        config, tmp_path):
    dest_dir = tmp_path / "cache"
    dest_dir.mkdir()
    item = {"public_id": "abc123", "format": "jpg"}   # thieu version
    assert sa.download(item, dest_dir) is None         # khong duoc nem KeyError


# ---- Phase 3: list -> pick random -> download -> verify ------------------

def _jpeg_bytes(size=(40, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "green").save(buffer, "JPEG")
    return buffer.getvalue()


def test_pick_random_is_seedable():
    items = [{"public_id": f"p{i}"} for i in range(20)]
    rng_a, rng_b = random.Random(42), random.Random(42)
    picks_a = [sa.pick_random(items, rng_a) for _ in range(10)]
    picks_b = [sa.pick_random(items, rng_b) for _ in range(10)]
    assert picks_a == picks_b


def test_verify_photo_accepts_a_small_real_image(tmp_path):
    path = tmp_path / "ok.jpg"
    path.write_bytes(_jpeg_bytes((100, 80)))
    assert sa.verify_photo(path) is True


def test_verify_photo_rejects_garbage_bytes(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not a jpeg at all")
    assert sa.verify_photo(path) is False


def test_verify_photo_rejects_oversized_image(tmp_path):
    path = tmp_path / "huge.jpg"
    path.write_bytes(_jpeg_bytes((100, 80)))
    assert sa.verify_photo(path, max_side=50) is False


def test_pick_random_and_download_empty_list_is_a_distinct_error(config, monkeypatch, tmp_path):
    monkeypatch.setattr(sa.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(
                            json.dumps({"resources": []}).encode("utf-8")))
    assert sa.pick_random_and_download(tmp_path / "cache") == sa.EMPTY_ALBUM


def test_pick_random_and_download_no_network_is_a_distinct_error(config, monkeypatch,
                                                                  tmp_path):
    def boom(*a, **k):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(sa.urllib.request, "urlopen", boom)
    assert sa.pick_random_and_download(tmp_path / "cache") == sa.NO_CONNECTION


def test_pick_random_and_download_success_selects_and_caches_one_item(
        config, monkeypatch, tmp_path):
    resources = [{"public_id": "a", "version": "1", "format": "jpg"},
                {"public_id": "b", "version": "1", "format": "jpg"}]
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(json.dumps({"resources": resources}).encode("utf-8"))
        return _FakeResponse(_jpeg_bytes())

    monkeypatch.setattr(sa.urllib.request, "urlopen", fake_urlopen)
    dest_dir = tmp_path / "cache"
    result = sa.pick_random_and_download(dest_dir, rng=random.Random(1))
    assert isinstance(result, Path)
    assert result.parent.resolve() == dest_dir.resolve()
    assert result.read_bytes() == _jpeg_bytes()


def test_pick_random_and_download_rejects_a_bad_image_and_cleans_up(
        config, monkeypatch, tmp_path):
    resources = [{"public_id": "a", "version": "1", "format": "jpg"}]
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(json.dumps({"resources": resources}).encode("utf-8"))
        return _FakeResponse(b"not actually a jpeg")

    monkeypatch.setattr(sa.urllib.request, "urlopen", fake_urlopen)
    dest_dir = tmp_path / "cache"
    result = sa.pick_random_and_download(dest_dir, rng=random.Random(1))
    assert result == sa.NO_CONNECTION
    assert list(dest_dir.iterdir()) == []   # tap xau khong duoc o lai trong cache
