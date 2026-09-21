"""Developer credentials must never turn ordinary pytest into a paid run."""

import socket

import pytest


@pytest.mark.parametrize("host", ["api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com"])
def test_external_dns_is_blocked_even_with_provider_credentials(monkeypatch, host):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "fake-offline-regression-key")
    with pytest.raises(RuntimeError, match="Offline test suite"):
        socket.getaddrinfo(host, 443)


@pytest.mark.parametrize("operation", ["connect", "connect_ex", "sendto"])
def test_external_ip_dispatch_is_blocked_before_network_io(operation):
    kind = socket.SOCK_DGRAM if operation == "sendto" else socket.SOCK_STREAM
    with socket.socket(socket.AF_INET, kind) as client:
        with pytest.raises(RuntimeError, match="Offline test suite"):
            if operation == "sendto":
                client.sendto(b"offline", ("192.0.2.1", 443))
            else:
                getattr(client, operation)(("192.0.2.1", 443))


def test_local_wire_fixtures_can_exchange_bytes():
    with socket.socket() as server, socket.socket() as client:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client.settimeout(2)
        client.connect(server.getsockname())
        connection, _ = server.accept()
        with connection:
            connection.settimeout(2)
            client.sendall(b"local fixture")
            assert connection.recv(64) == b"local fixture"
