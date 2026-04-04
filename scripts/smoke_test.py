import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from client import encode_message, decode_message
from clients.vanilla_client import encode_plain_message, decode_plain_message
from peers.base import ReplayProtector


def main():
    print("Running smoke_test: encode/decode and replay protection checks...")
    x = np.random.randn(4, 8).astype(np.float32)

    print("- Testing encrypted message encode/decode")
    blob = encode_message("FWD_REQ", "sess", "cli_x", x, "secret", pad_multiple=64)
    op, sess, sender, tensor, header = decode_message(blob, "secret")
    assert op == "FWD_REQ" and sess == "sess" and sender == "cli_x"
    assert np.allclose(tensor, x)
    assert "msg_id" in header and header.get("nonce")
    print("  encrypted message encode/decode OK")

    print("- Testing tamper detection")
    tampered = blob[:-1] + bytes([blob[-1] ^ 0x01])
    try:
        decode_message(tampered, "secret")
        raise AssertionError("tampered ciphertext should fail")
    except Exception:
        print("  tamper detection OK")

    print("- Testing plaintext message encode/decode")
    plain = encode_plain_message("INFER_REQ", "s2", "cli_y", x)
    op2, sess2, sender2, tensor2, header2 = decode_plain_message(plain)
    assert op2 == "INFER_REQ" and sess2 == "s2" and sender2 == "cli_y"
    assert np.allclose(tensor2, x)
    assert "msg_id" in header2
    print("  plaintext message encode/decode OK")

    print("- Testing replay protector")
    rp = ReplayProtector(3)
    mid = header["msg_id"]
    assert rp.seen_or_add(mid) is False
    assert rp.seen_or_add(mid) is True
    print("  replay protector OK")

    print("smoke ok: all tests passed")


if __name__ == "__main__":
    main()