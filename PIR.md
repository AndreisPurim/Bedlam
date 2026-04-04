# Toy 1-server PIR (computational PIR) with OpenFHE-Python (BFV batching).
# - Server stores DB as a packed plaintext vector (length = batch_size).
# - Client sends Enc(one_hot(index)).
# - Server computes Enc(one_hot) * PT(db) and then sums all slots via rotations.
#
# WARNING: This is a minimal demo, not production PIR.

from openfhe import *


def setup_bfv_context(batch_size: int, plaintext_modulus: int = 65537, mult_depth: int = 2):
    params = CCParamsBFVRNS()
    params.SetPlaintextModulus(plaintext_modulus)
    params.SetMultiplicativeDepth(mult_depth)
    # BFV batching slots are controlled via SetBatchSize in the Python wrapper examples
    params.SetBatchSize(batch_size)

    cc = GenCryptoContext(params)

    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    return cc


def gen_rotation_steps(n: int):
    # power-of-two steps for a log(n) reduction
    steps = []
    step = 1
    while step < n:
        steps.append(step)
        step *= 2
    return steps


def client_make_query(cc, public_key, index: int, n: int):
    sel = [0] * n
    sel[index] = 1
    pt = cc.MakePackedPlaintext(sel)
    return cc.Encrypt(public_key, pt)


def server_answer_query(cc, query_ct, db_vector, rotation_steps):
    """
    db_vector: list[int] of length n (server-side database)
    Returns: ciphertext encrypting db_vector[index] replicated in all slots.
    """
    db_pt = cc.MakePackedPlaintext(db_vector)

    # Elementwise multiply: one-hot * db
    ct = cc.EvalMult(query_ct, db_pt)

    # Sum all slots via rotate-and-add reduction
    for s in rotation_steps:
        ct = cc.EvalAdd(ct, cc.EvalRotate(ct, s))
    return ct


def client_decrypt_answer(cc, secret_key, answer_ct, plaintext_modulus: int, n: int):
    pt = cc.Decrypt(answer_ct, secret_key)
    pt.SetLength(n)
    vals = pt.GetPackedValue()

    # OpenFHE often returns centered reps in [-p/2, p/2); map back to [0, p)
    v0 = vals[0]
    if v0 < 0:
        v0 += plaintext_modulus
    return v0


def demo():
    # --- Demo database (server-side) ---
    n = 16  # must be <= batching capacity for chosen parameters
    db = [10 * i + 3 for i in range(n)]  # e.g., [3, 13, 23, ...]
    target_index = 7

    plaintext_modulus = 65537

    # --- Setup ---
    cc = setup_bfv_context(batch_size=n, plaintext_modulus=plaintext_modulus, mult_depth=2)
    print("Ring dimension:", cc.GetRingDimension())

    kp = cc.KeyGen()

    # Relinearization key (safe to generate even if you only do ct*pt once)
    cc.EvalMultKeyGen(kp.secretKey)

    # Rotation keys needed for EvalRotate in the slot-sum reduction
    steps = gen_rotation_steps(n)
    cc.EvalRotateKeyGen(kp.secretKey, steps)

    # --- Client: make encrypted query ---
    query_ct = client_make_query(cc, kp.publicKey, target_index, n)

    # --- Server: compute encrypted answer ---
    answer_ct = server_answer_query(cc, query_ct, db, steps)

    # --- Client: decrypt ---
    got = client_decrypt_answer(cc, kp.secretKey, answer_ct, plaintext_modulus, n)

    print("DB[target_index] =", db[target_index])
    print("PIR result       =", got)


if __name__ == "__main__":
    demo()
