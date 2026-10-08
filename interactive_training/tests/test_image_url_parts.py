import base64
import io
import os
import socket
import subprocess
import sys
import urllib.error
from unittest.mock import Mock

import pytest

from interactive_training.image_types import ImageChunk
from interactive_training.renderers import base as renderer_base
from interactive_training.renderers.base import (
    Message,
    ToolCall,
    _PinnedHTTPConnection,
    _PinnedHTTPSConnection,
    _SafeImageRedirectHandler,
    _SafeImageHTTPSHandler,
    RenderContext,
    _load_image_reference,
    image_to_chunk,
)
from interactive_training.renderers.qwen3 import Qwen3VLRenderer

Image = pytest.importorskip("PIL.Image")


class ImageProcessor:
    merge_size = 1

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        assert (height, width) == (2, 3)
        return 6


class Tokenizer:
    def __init__(self):
        self.tokens = {}

    def encode(self, text, add_special_tokens=True):
        return [self.tokens.setdefault(text, len(self.tokens) + 1)]


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def _animated_png_bytes() -> bytes:
    buffer = io.BytesIO()
    first = Image.new("RGBA", (3, 2), color="red")
    second = Image.new("RGBA", (3, 2), color="blue")
    first.save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


def test_image_to_chunk_decodes_base64_data_uri():
    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()

    chunk = image_to_chunk(data_uri, ImageProcessor())

    assert chunk.format == "jpeg"
    assert chunk.data.startswith(b"\xff\xd8")
    assert chunk.expected_tokens == 6


def test_image_to_chunk_downloads_public_url(monkeypatch):
    monkeypatch.setattr(
        "interactive_training.renderers.base._load_image_reference",
        lambda url: _png_bytes(),
    )

    chunk = image_to_chunk("https://example.test/part.png", ImageProcessor())

    assert chunk.format == "jpeg"
    assert chunk.data.startswith(b"\xff\xd8")
    assert chunk.expected_tokens == 6


def test_image_loader_downloads_public_url(monkeypatch):
    image_data = _png_bytes()
    captured = {}

    class FakeResponse:
        headers = {"Content-Length": str(len(image_data))}

        def read(self, size):
            captured["read_size"] = size
            return image_data

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["user_agent"] = request.get_header("User-agent")
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(
        renderer_base.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
        ],
    )
    monkeypatch.setattr(
        renderer_base.urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    assert _load_image_reference("https://example.test/part.png") == image_data
    assert captured["url"] == "https://example.test/part.png"
    assert captured["user_agent"] == "Interactive TrainingImageLoader/1.0"
    assert 0 < captured["timeout"] <= renderer_base.IMAGE_FETCH_TIMEOUT_SECONDS
    assert captured["read_size"] == renderer_base._IMAGE_READ_CHUNK_BYTES


def test_image_loader_reuses_owner_only_cache_without_storing_url(
    monkeypatch, tmp_path
):
    image_data = _png_bytes()
    calls = []
    url = "https://example.test/part.png?sig=TOPSECRET"
    monkeypatch.setattr(
        renderer_base,
        "_download_image_reference",
        lambda reference: calls.append(reference) or image_data,
    )

    assert _load_image_reference(url, cache_dir=str(tmp_path)) == image_data
    monkeypatch.setattr(
        renderer_base,
        "_download_image_reference",
        lambda _reference: pytest.fail("cache hit attempted a network download"),
    )
    assert _load_image_reference(url, cache_dir=str(tmp_path)) == image_data

    cache_files = list(tmp_path.iterdir())
    assert len(cache_files) == 1
    assert "TOPSECRET" not in cache_files[0].name
    # POSIX mode bits do not represent Windows ACL permissions.
    if os.name == "posix":
        assert cache_files[0].stat().st_mode & 0o777 == 0o600
    assert calls == [url]


def test_image_loader_does_not_alias_distinct_signed_urls(monkeypatch, tmp_path):
    first_url = "https://example.test/part.png?sig=FIRST"
    second_url = "https://example.test/part.png?sig=SECOND"
    downloads = []
    monkeypatch.setattr(
        renderer_base,
        "_download_image_reference",
        lambda reference: downloads.append(reference) or _png_bytes(),
    )

    _load_image_reference(first_url, cache_dir=str(tmp_path))
    _load_image_reference(second_url, cache_dir=str(tmp_path))

    assert downloads == [first_url, second_url]
    assert len(list(tmp_path.iterdir())) == 2


def test_image_loader_reuses_cache_across_process_restart(monkeypatch, tmp_path):
    image_data = _png_bytes()
    url = "https://example.test/part.png?sig=restart-test"
    monkeypatch.setattr(
        renderer_base,
        "_download_image_reference",
        lambda _reference: image_data,
    )
    assert _load_image_reference(url, cache_dir=str(tmp_path)) == image_data

    script = """
import base64
import sys
from interactive_training.renderers.base import _load_image_reference

data = _load_image_reference(sys.argv[1], cache_dir=sys.argv[2])
sys.stdout.write(base64.b64encode(data).decode("ascii"))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, url, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert base64.b64decode(result.stdout) == image_data


def test_image_download_failure_does_not_expose_signed_url(monkeypatch):
    signed_url = "https://example.test/part.png?sig=TOPSECRET"

    class FailingOpener:
        def open(self, request, timeout):
            raise urllib.error.URLError(f"failed fetching {request.full_url}")

    monkeypatch.setattr(renderer_base, "_validate_public_image_url", lambda _url: None)
    monkeypatch.setattr(
        renderer_base.urllib.request,
        "build_opener",
        lambda *_handlers: FailingOpener(),
    )
    monkeypatch.setattr(renderer_base.time, "sleep", lambda _seconds: None)

    with pytest.raises(ValueError) as exc_info:
        renderer_base._download_image_reference(signed_url)

    assert "example.test" in str(exc_info.value)
    assert "TOPSECRET" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_image_loader_enforces_one_deadline_across_trickled_reads(monkeypatch):
    clock = [100.0]
    socket_timeouts = []

    class FakeSocket:
        def settimeout(self, timeout):
            socket_timeouts.append(timeout)

    class FakeRaw:
        _sock = FakeSocket()

    class FakeFile:
        raw = FakeRaw()

    class TrickleResponse:
        headers = {}
        fp = FakeFile()

        def read1(self, _size):
            clock[0] += 6
            return b"x"

    monkeypatch.setattr(renderer_base.time, "monotonic", lambda: clock[0])

    with pytest.raises(ValueError, match="time limit"):
        renderer_base._read_bounded_image_response(
            TrickleResponse(), deadline=110.0
        )

    assert socket_timeouts == [10.0, 4.0]


def test_image_loader_rejects_oversized_chunked_response(monkeypatch):
    chunks = [b"123", b"45"]

    class ChunkedResponse:
        headers = {}

        def read(self, _size):
            return chunks.pop(0) if chunks else b""

    monkeypatch.setattr(renderer_base, "MAX_IMAGE_BYTES", 4)

    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        renderer_base._read_bounded_image_response(ChunkedResponse())


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_image_loader_retries_transient_http_failures(
    monkeypatch,
    status_code,
):
    image_data = _png_bytes()
    attempts = []
    sleeps = []

    class FakeResponse:
        headers = {"Content-Length": str(len(image_data))}

        def read(self, _size):
            return image_data

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class FakeOpener:
        def open(self, request, timeout):
            attempts.append((request.full_url, timeout))
            if len(attempts) == 1:
                raise urllib.error.HTTPError(
                    request.full_url, status_code, "retry", None, None,
                )
            return FakeResponse()

    monkeypatch.setattr(renderer_base, "_validate_public_image_url", lambda _url: None)
    monkeypatch.setattr(
        renderer_base.urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )
    monkeypatch.setattr(renderer_base.time, "sleep", sleeps.append)

    assert renderer_base._download_image_reference(
        "https://example.test/part.png"
    ) == image_data
    assert len(attempts) == 2
    assert sleeps == [0.1]


def test_image_loader_does_not_retry_terminal_http_failure(monkeypatch):
    attempts = []

    class FakeOpener:
        def open(self, request, timeout):
            attempts.append((request.full_url, timeout))
            raise urllib.error.HTTPError(
                request.full_url, 404, "missing", None, None,
            )

    monkeypatch.setattr(renderer_base, "_validate_public_image_url", lambda _url: None)
    monkeypatch.setattr(
        renderer_base.urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(ValueError, match="after 1 attempt.*HTTP 404"):
        renderer_base._download_image_reference(
            "https://example.test/part.png"
        )
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:password@example.com/image.png",
        "http://example.com/image.png",
        "http://localhost/image.png",
        "http://127.0.0.1/image.png",
        "http://169.254.169.254/metadata/instance",
        "http://[::1]/image.png",
    ],
)
def test_image_loader_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        _load_image_reference(url)


def test_image_connection_uses_validated_ip(monkeypatch):
    calls = []
    fake_socket = Mock()
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda address, *args, **kwargs: calls.append(address) or fake_socket,
    )

    connection = _PinnedHTTPConnection(
        "example.test", pinned_ip="203.0.113.10"
    )
    connection.connect()

    assert calls == [("203.0.113.10", 80)]
    assert connection.sock is fake_socket


def test_image_connection_deadline_closes_socket(monkeypatch):
    fake_socket = Mock()
    timers = []

    class FakeTimer:
        def __init__(self, interval, function):
            self.interval = interval
            self.function = function
            self.daemon = False
            self.started = False
            self.cancelled = False
            timers.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(socket, "create_connection", Mock(return_value=fake_socket))
    monkeypatch.setattr(renderer_base.threading, "Timer", FakeTimer)
    monkeypatch.setattr(renderer_base.time, "monotonic", lambda: 100.0)

    connection = _PinnedHTTPConnection(
        "example.test", pinned_ip="8.8.8.8", deadline=110.0
    )
    connection.connect()

    assert len(timers) == 1
    assert timers[0].interval == 10.0
    assert timers[0].daemon
    assert timers[0].started

    timers[0].function()

    assert timers[0].cancelled
    fake_socket.shutdown.assert_called_once_with(socket.SHUT_RDWR)
    fake_socket.close.assert_called_once_with()
    assert connection.sock is None


def test_https_connection_pins_ip_but_verifies_original_hostname(monkeypatch):
    raw_socket = Mock()
    tls_socket = Mock()
    tls_context = Mock()
    tls_context.wrap_socket.return_value = tls_socket
    create_connection = Mock(return_value=raw_socket)
    monkeypatch.setattr(socket, "create_connection", create_connection)

    connection = _PinnedHTTPSConnection(
        "example.test", 8443, pinned_ip="8.8.8.8", context=tls_context
    )
    connection.connect()

    assert create_connection.call_args.args[0] == ("8.8.8.8", 8443)
    tls_context.wrap_socket.assert_called_once_with(
        raw_socket, server_hostname="example.test"
    )
    assert connection.sock is tls_socket


def test_safe_https_handler_uses_its_tls_context(monkeypatch):
    handler = _SafeImageHTTPSHandler()
    captured = {}
    monkeypatch.setattr(
        renderer_base,
        "_resolve_public_image_url",
        lambda _url: "8.8.8.8",
    )
    monkeypatch.setattr(
        handler,
        "do_open",
        lambda connection_factory, request, **kwargs: captured.update(
            connection_factory=connection_factory,
            request=request,
            kwargs=kwargs,
        ),
    )
    request = renderer_base.urllib.request.Request("https://example.test/image.png")

    handler.https_open(request)

    assert captured["request"] is request
    assert captured["kwargs"] == {"context": handler._context}


def test_image_redirect_rejects_https_downgrade():
    request = renderer_base.urllib.request.Request(
        "https://example.test/image.png"
    )

    with pytest.raises(ValueError, match="must not downgrade"):
        _SafeImageRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://example.test/image.png",
        )


def test_image_loader_rejects_mixed_public_and_private_dns(monkeypatch):
    monkeypatch.setattr(
        renderer_base.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )

    with pytest.raises(ValueError, match="public unicast"):
        _load_image_reference("https://example.test/image.png")


def test_image_loader_rejects_multicast_address(monkeypatch):
    monkeypatch.setattr(
        renderer_base.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("224.0.0.1", 443))
        ],
    )

    with pytest.raises(ValueError, match="public unicast"):
        _load_image_reference("https://example.test/image.png")


@pytest.mark.parametrize(
    "translated_address",
    [
        "64:ff9b::127.0.0.1",
        "64:ff9b::169.254.169.254",
        "64:ff9b:1::808:808",
    ],
)
def test_image_loader_rejects_unsafe_nat64_address(
    monkeypatch, translated_address
):
    monkeypatch.setattr(
        renderer_base.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                (translated_address, 443, 0, 0),
            )
        ],
    )

    with pytest.raises(ValueError, match="public unicast"):
        _load_image_reference("https://example.test/image.png")


def test_image_loader_rejects_invalid_data_uri_base64():
    with pytest.raises(ValueError, match="invalid base64"):
        _load_image_reference("data:image/png;base64,not-base64")


def test_image_to_chunk_rejects_malformed_image_data():
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nnot an image").decode()

    with pytest.raises(ValueError, match="malformed or unsafe"):
        image_to_chunk(f"data:image/png;base64,{encoded}", ImageProcessor())


def test_image_to_chunk_rejects_animated_png():
    encoded = base64.b64encode(_animated_png_bytes()).decode()

    with pytest.raises(ValueError, match="Animated images are not supported"):
        image_to_chunk(f"data:image/png;base64,{encoded}", ImageProcessor())


def test_image_to_chunk_rejects_pillow_decompression_bomb(monkeypatch):
    encoded = base64.b64encode(_png_bytes()).decode()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(ValueError, match="malformed or unsafe"):
        image_to_chunk(f"data:image/png;base64,{encoded}", ImageProcessor())


def test_image_to_chunk_rejects_disallowed_format_before_pillow(monkeypatch):
    encoded = base64.b64encode(b"GIF89a\x01\x00\x01\x00").decode()
    image_open = Mock()
    monkeypatch.setattr(Image, "open", image_open)

    with pytest.raises(ValueError, match="JPEG, PNG, or WebP"):
        image_to_chunk(f"data:image/png;base64,{encoded}", ImageProcessor())

    image_open.assert_not_called()


def test_image_loader_rejects_oversized_data(monkeypatch):
    monkeypatch.setattr("interactive_training.renderers.base.MAX_IMAGE_BYTES", 4)
    encoded = base64.b64encode(b"12345").decode()

    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        _load_image_reference(f"data:image/png;base64,{encoded}")


def test_image_to_chunk_rejects_excessive_pixels_and_nonpositive_tokens(monkeypatch):
    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    monkeypatch.setattr("interactive_training.renderers.base.MAX_IMAGE_PIXELS", 5)
    with pytest.raises(ValueError, match="pixel limit"):
        image_to_chunk(data_uri, ImageProcessor())

    monkeypatch.setattr("interactive_training.renderers.base.MAX_IMAGE_PIXELS", 25_000_000)

    class InvalidTokenProcessor(ImageProcessor):
        def get_number_of_image_patches(self, height, width, images_kwargs=None):
            return 0

    with pytest.raises(ValueError, match="must be positive"):
        image_to_chunk(data_uri, InvalidTokenProcessor())

    class LargeTokenProcessor(ImageProcessor):
        def get_number_of_image_patches(self, height, width, images_kwargs=None):
            return 4097

    assert image_to_chunk(data_uri, LargeTokenProcessor()).expected_tokens == 4097


@pytest.mark.parametrize(
    "renderer",
    [
        Qwen3VLRenderer(Tokenizer(), object()),
    ],
)
def test_vision_renderers_accept_foundry_image_url(monkeypatch, renderer):
    url = "data:image/png;base64,iVBORw0KGgo="
    received_images = []
    image_chunk = ImageChunk(
        data=b"\xff\xd8\xffjpeg", format="jpeg", expected_tokens=6
    )
    monkeypatch.setattr(
        "interactive_training.renderers.base.image_to_chunk",
        lambda image_or_str, image_processor: (
            received_images.append(image_or_str) or image_chunk
        ),
    )
    message = Message(
        role="user",
        content=[
            {
                "type": "image_url",
                "image_url": {"url": url, "detail": "high"},
            }
        ],
    )

    rendered = renderer.render_message(
        message,
        RenderContext(idx=0, is_last=True),
    )

    assert image_chunk in rendered.output
    assert received_images == [url]


def test_to_openai_message_preserves_foundry_image_url():
    renderer = Qwen3VLRenderer(Tokenizer(), object())
    content = [
        {"type": "text", "text": "What defect is present?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.test/part.jpg",
                "detail": "high",
            },
        },
    ]

    result = renderer.to_openai_message(Message(role="user", content=content))

    assert result["content"] == content


def test_qwen3_vl_renders_image_url_between_vision_markers(monkeypatch):
    tokenizer = Tokenizer()
    renderer = Qwen3VLRenderer(tokenizer, object())
    image_chunk = ImageChunk(
        data=b"\xff\xd8\xffjpeg", format="jpeg", expected_tokens=6
    )
    monkeypatch.setattr(
        "interactive_training.renderers.base.image_to_chunk",
        lambda image_or_str, image_processor: image_chunk,
    )
    message = Message(
        role="user",
        content=[
            {"type": "text", "text": "before"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/part.jpg"},
            },
            {"type": "text", "text": "after"},
        ],
    )

    rendered = renderer.render_message(
        message,
        RenderContext(idx=0, is_last=True),
    )

    assert rendered.output == [
        renderer_base.ModelInputChunk(
            tokens=tokenizer.encode(
                "before<|vision_start|>", add_special_tokens=False
            )
        ),
        image_chunk,
        renderer_base.ModelInputChunk(
            tokens=tokenizer.encode(
                "<|vision_end|>after<|im_end|>", add_special_tokens=False
            )
        ),
    ]


def test_qwen3_openai_image_message_preserves_reasoning_and_tool_calls():
    renderer = Qwen3VLRenderer(Tokenizer(), object())
    tool_call = ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(name="inspect", arguments='{"id": 7}'),
    )
    message = Message(
        role="assistant",
        content=[
            {"type": "thinking", "thinking": "check the image"},
            {"type": "text", "text": "Found it."},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/part.jpg"},
            },
        ],
        tool_calls=[tool_call],
    )

    result = renderer.to_openai_message(message)

    assert result["content"] == [
        {"type": "text", "text": "Found it."},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/part.jpg"},
        },
    ]
    assert result["reasoning_content"] == "check the image"
    assert result["tool_calls"] == [
        {
            "type": "function",
            "id": "call-1",
            "function": {"name": "inspect", "arguments": '{"id": 7}'},
        }
    ]