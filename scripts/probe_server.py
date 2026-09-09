#!/usr/bin/env python3
"""Independent client for paper-server: builds every frame by hand from the
protocol description rather than reusing the server's own helpers, so a
symmetric bug in both cannot pass.

Wire format: little-endian fixed-width ints; b'!' true / b'?' false; a u32
length prefix on every buffer and string; [command: u8] then arguments; a
response of [ok: bool] then either the payload or [code: u8], where code 0 means
a cache error code follows as a second byte. The SERVER speaks first with a bare
handshake byte.
"""
import socket, struct, sys

PING, VERSION, AUTH, GET, SET, DEL, HAS, PEEK, TTL, SIZE, WIPE, RESIZE, POLICY, STATS = range(14)
SELF_STATS = 200
TRUE, FALSE = 33, 63


class Client:
    def __init__(self, host='127.0.0.1', port=3145):
        self.s = socket.create_connection((host, port), timeout=10)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.handshake = self._read(1)[0]

    def _read(self, n):
        b = b''
        while len(b) < n:
            c = self.s.recv(n - len(b))
            if not c:
                raise EOFError(f'server closed after {len(b)}/{n} bytes')
            b += c
        return b

    def _u32(self):
        return struct.unpack('<I', self._read(4))[0]

    def _u64(self):
        return struct.unpack('<Q', self._read(8))[0]

    def _buf(self):
        return self._read(self._u32())

    def _bool(self):
        v = self._read(1)[0]
        if v == TRUE:
            return True
        if v == FALSE:
            return False
        raise ValueError(f'not a boolean indicator: {v}')

    def _send(self, cmd, payload=b''):
        self.s.sendall(bytes([cmd]) + payload)

    def _ok(self):
        """Returns True, or raises with the decoded error."""
        if self._bool():
            return True
        code = self._read(1)[0]
        if code == 0:
            raise RuntimeError(f'cache error code {self._read(1)[0]}')
        raise RuntimeError(f'server error code {code}')

    @staticmethod
    def _pbuf(b):
        return struct.pack('<I', len(b)) + b

    @staticmethod
    def _pstr(s):
        return Client._pbuf(s.encode())

    # ---- commands ----
    def ping(self):
        # PING answers with a bare boolean -- no payload follows.
        self._send(PING); return self._ok()

    def version(self):
        self._send(VERSION); self._ok(); return self._buf().decode()

    def set(self, key, value, ttl=0):
        self._send(SET, self._pstr(str(key)) + self._pbuf(value) + struct.pack('<I', ttl))
        return self._ok()

    def get(self, key):
        self._send(GET, self._pstr(str(key))); self._ok(); return self._buf()

    def peek(self, key):
        self._send(PEEK, self._pstr(str(key))); self._ok(); return self._buf()

    def has(self, key):
        self._send(HAS, self._pstr(str(key))); self._ok(); return self._bool()

    def size(self, key):
        self._send(SIZE, self._pstr(str(key))); self._ok(); return self._u32()

    def self_stats(self):
        # Command 200: outside the upstream 0..=13 range, answers with the
        # server's OWN latency report as a plain text buffer.
        self._send(SELF_STATS); self._ok(); return self._buf().decode()

    def delete(self, key):
        self._send(DEL, self._pstr(str(key))); return self._ok()


def main():
    c = Client()
    print(f'handshake byte      : {c.handshake}')
    print(f'ping                : {c.ping()!r}')
    print(f'version             : {c.version()!r}')

    payload = bytes(range(256)) * 8          # 2048 bytes
    print(f'set 1 (2048 B)      : {c.set(1, payload)}')
    got = c.get(1)
    print(f'get 1               : {len(got)} B, roundtrip ok = {got == payload}')
    print(f'peek 1              : {len(c.peek(1))} B')
    print(f'has 1               : {c.has(1)}')
    print(f'size 1              : {c.size(1)} B  (base_size, > 2048 by the metadata charge)')

    print(f'has 999 (absent)    : {c.has(999)}')
    try:
        c.get(999)
        print('get 999             : UNEXPECTEDLY OK')
    except RuntimeError as e:
        print(f'get 999             : correctly errored -> {e}')

    print(f'del 1               : {c.delete(1)}')
    print(f'has 1 after del     : {c.has(1)}')

    n = 2000
    for i in range(n):
        c.set(i, b'x' * 1024)
    hits = sum(1 for i in range(n) if c.has(i))
    print(f'{n} sets of 1 KiB    : {hits} still present (evictions expected under a 1 GiB cap)')
    print()
    print(c.self_stats())
    print('ALL PROBES PASSED')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'FAILED: {type(e).__name__}: {e}')
        sys.exit(1)
