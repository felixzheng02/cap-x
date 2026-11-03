"""Wire protocol shared by the simulator server and the CaP-X client.

This file is deliberately **duplicated** on both sides rather than imported, so the two
environments stay dependency-isolated. Keep it small and keep the copies identical; the
``PROTOCOL_VERSION`` handshake exists to catch drift loudly instead of silently.

Transport: ZeroMQ DEALER (client) <-> ROUTER (server), msgpack frames.

DEALER/ROUTER rather than REQ/REP on purpose: REQ is a strict lock-step state machine, so a
single timeout leaves the socket unusable. The CaP-X harness enforces a trial timeout with
SIGALRM -- which cannot reach another process -- and retries up to 3x, so timeouts are expected
and must not poison the connection.

Request:   {"id": int, "method": str, "params": dict}
Response:  {"id": int, "ok": bool, "result": any}  |  {"id": int, "ok": False, "error": str}
"""

from __future__ import annotations

PROTOCOL_VERSION = 1

# --- methods ---------------------------------------------------------------
PING = "ping"                       # {} -> {"pong": True, "protocol": int}
GET_SPEC = "get_spec"               # {} -> static robot spec; fetched once at connect
RESET = "reset"                     # {"seed": int|None} -> proprio + rgb + episode metadata
STEP = "step"                       # {"action": [float]*action_dim} -> proprio + rgb
GET_DEPTH = "get_depth"             # {"camera": str|None} -> {"depth": (H,W) float32 metres}
GET_SEGMENTATION = "get_segmentation"   # {"camera": str|None} -> {"body_ids", "id_to_name"}
RENDER = "render"                   # {"camera": str|None} -> {"rgb": (H,W,3) uint8} fresh render
EVALUATE = "evaluate"               # {} -> {"reward", "success", "instruction"}   SIM-ONLY
VIDEO_COUNT = "video_frame_count"   # {} -> {"count": int}
VIDEO_ENABLE = "video_enable"       # {"enabled": bool, "clear": bool, "wrist": bool}
VIDEO_DUMP = "video_dump"           # {"start": int, "end": int, "path": str} -> /dev/shm handoff
CLOSE = "close"                     # {} -> {"bye": True}

ALL_METHODS = {
    PING, GET_SPEC, RESET, STEP, GET_DEPTH, GET_SEGMENTATION,
    RENDER, EVALUATE, VIDEO_COUNT, VIDEO_ENABLE, VIDEO_DUMP, CLOSE,
}

DEFAULT_ENDPOINT = "ipc:///tmp/capx-sim-0.sock"


def pack(obj) -> bytes:
    import msgpack
    import msgpack_numpy

    return msgpack.packb(obj, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(buf: bytes):
    import msgpack
    import msgpack_numpy

    # strict_map_key=False: segmentation maps body *ids* (ints) to names, and msgpack
    # rejects non-str/bytes map keys by default.
    return msgpack.unpackb(
        buf, object_hook=msgpack_numpy.decode, raw=False, strict_map_key=False
    )
