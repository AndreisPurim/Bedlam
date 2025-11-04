# OMRsplit

Oblivious Message Retrieval applied to Split-Learning

------------

## 1. What we’ve built

### Split-learning setup

* We’re using **MNIST** as the toy problem.
* Each client has:

  * **M1**: early feature extractor.
  * **M3**: classifier head.
* Separate peers hold:

  * **M2**: middle model.

The training loop is standard split learning:

1. Client runs `x → M1 → z_cut`.
2. Client sends `z_cut` to an M2 peer.
3. M2 runs `z_cut → M2 → z_mid`, sends `z_mid` back.
4. Client runs `z_mid → M3 → logits`, computes loss and gradients:

   * updates M3 locally,
   * sends `dL/dz_mid` back to M2.
5. M2 backprops through M2, sends `dL/dz_cut` to client.
6. Client backprops through M1.

We do this with multiple clients and multiple M2 peers, using **Ray** actors.

---

### Board and peers

* There is a central **Board** actor.

* The Board:

  * only stores messages per `receiver`,
  * each message has: `msg_id`, `sender`, `receiver`, `payload`, `timestamp`,
  * it does **not** know:

    * if a message is forward, backward, or inference,
    * anything about sessions,
    * anything about tensors.

* **PeerM1M3** (clients):

  * hold M1, M3, and their data shard.
  * talk **only** to the Board.
  * know which M2 they want (`target_m2` from config).
  * run the full training loop by posting and polling messages on the Board.

* **PeerM2**:

  * holds M2 and its optimizer.
  * loops over:

    * polling messages from Board,
    * decrypting,
    * checking the `op` inside the payload (`FWD_REQ`, `BWD_REQ`, `INFER_REQ`),
    * doing the right thing and responding via the Board.

All peers, the Board, and run options (epochs, batch size, etc.) are controlled by `config.yaml`.

---

### Logging & metrics

* There’s a **global log** (`global.log`) in a run directory like `runs/run_YYYYMMDD_HHMMSS`.
* Each peer (Board, every M1M3 client, every M2) has its own log file.
* Each client writes a CSV with per-step metrics (`epoch, step, loss, acc`).

So you can see:

* how training progresses for each client,
* what the M2 peers are doing when `verbose` is on,
* what the Board is seeing (in a limited way).

---

## 2. Privacy / “OMR-ish” properties

### Encrypted envelopes

Every message between clients and M2 goes through the Board as an **opaque byte blob**.

Inside that blob (before encryption) we have:

* a header:

  ```json
  {
    "op": "FWD_REQ" | "FWD_RES" | "BWD_REQ" | "BWD_RES" | "INFER_REQ" | "INFER_RES",
    "session": "<random session id>",
    "sender": "<client pseudonym or M2 name>",
    "tensor_len": <length of serialized tensor>
  }
  ```
* the serialized tensor.
* random padding to reach a fixed multiple of bytes (e.g., 1024).

Then we encrypt the whole thing with a stream cipher derived from the shared key in `config.yaml`:

* The Board cannot read:

  * `op`,
  * `session`,
  * who the real client is,
  * tensors or their exact lengths.

It only knows:

* “something from X to Y, of size ≈ this many KB, at this time”.

---

### No “kind” or session id at the Board level

* The Board **does not store** any explicit:

  * message type,
  * forward/backward flag,
  * session id.

It just routes messages by `receiver`.

All protocol logic lives inside the encrypted header and is handled by the peers.

---

### Pseudonyms for clients (single-blind)

* Each M1M3 client generates a random pseudonym for the run:

  * e.g. `cli_83fa2c`.
* The **Board** and **M2** only see that pseudonym, not `"client_1"`.

Flow:

* Client → M2:

  * `sender = client_pseudonym`,
  * `receiver = "m2_a"`.
* M2 → client:

  * M2 reads `sender_pseudo` from header,
  * sends response with `receiver = client_pseudonym`.

So:

* M2 knows it is talking to some stable pseudonym,
* but not which concrete client instance that is (`client_1`, `client_2`, etc.).
* The Board only sees pseudonyms and M2 names, plus timestamps and sizes.

This matches your “single-blind” idea:

* client knows which M2 it’s using,
* M2 doesn’t know the real client identity.

---

### Padding

We pad the plaintext (header + tensor) to a fixed multiple of bytes before encrypting.

This means:

* the Board doesn’t see the exact tensor length,
* only a rounded-up size (e.g., 1024, 2048, 3072 bytes, etc.).

So side-channel info about exact feature size per message is reduced.

---

## 3. What is still leaking / missing

We’re not “fully OMR” yet. Some gaps:

1. **Crypto is still PoC-level**

   * It uses a SHA-256-based stream cipher we wrote.
   * Better than trivial XOR, but not a standard scheme with nonces and integrity.
   * For production you’d want AES-GCM or ChaCha20-Poly1305.

2. **Board still has metadata**

   * It sees sender and receiver pseudonyms and timestamps.
   * It can build a communication graph and timing profile:
     “pseudonym A sends to m2_a, then m2_a responds to A”, etc.

3. **M2 sees a stable pseudonym per client per run**

   * It doesn’t know `"client_1"`, but it knows “this same pseudonym sent many messages”.
   * For more anonymity, you might want per-session or per-message pseudonyms.

4. **No traffic shaping**

   * No dummy messages.
   * No batching.
   * No random delays.
   * Timing and volume patterns are visible.

So right now we have:

* payload confidentiality,
* metadata reduction,
* single-blind identity toward M2,
* but not yet strong protection against traffic analysis or full unlinkability.

---

## 4. TODO list (next clear steps)

Here’s a concrete TODO list, roughly in order of increasing difficulty.

### A. Crypto cleanup

* [ ] Replace the custom stream cipher with a proper AEAD (AES-GCM or ChaCha20-Poly1305) once environment issues are sorted.

  * Keep the same envelope format (header + tensor + padding).
  * Just swap `_crypt_bytes` for a library call.
* [ ] Add integrity/authentication:

  * Right now, there’s no MAC. With AEAD, we’d get this “for free”.

### B. Pseudonym strategy

* [ ] Decide the pseudonym scope:

  * keep current **per-client-per-run** pseudonym, or
  * move to **per-session pseudonyms** (more privacy, more complexity).
* [ ] If you choose per-session:

  * generate a fresh pseudonym per session,
  * let M2 only see per-session tokens, not a stable client identifier.

### C. Metadata minimization at the Board

* [ ] Consider hiding sender/receiver more:

  * e.g. Board only sees roles like “M2_pool” instead of specific names, and the M2 pool does some internal routing.
* [ ] Add simple stats logging:

  * number of messages per pseudonym,
  * total bytes per pseudonym,
    to see how much “pattern” we’re leaking.

### D. Traffic-shaping experiments

* [ ] Add optional **dummy messages**:

  * clients send occasional encrypted nonsense to M2,
  * M2 discards them after decrypting.
* [ ] Add optional **random delays**:

  * jitter before sending or polling to blur timing.
* [ ] Later: experiment with small **batching** of messages at the Board.

### E. UX/instrumentation

* [ ] Add a small summary script that:

  * reads the per-client CSV metrics,
  * plots accuracy/loss,
  * and maybe overlays communication volume (TX/RX MB per step).
* [ ] Log more “high-level” event summaries:

  * e.g. “client_1 (pseudonym cli_xxxx) trained 2 epochs with m2_a; final acc = 0.97”.

### F. Longer-term OMR-ish stuff

* [ ] Explore removing even more structural metadata:

  * e.g. make Board blind to M2 identity as well, using tags or pools.
* [ ] Think about how homomorphic operations or secure aggregation might fit:

  * Board or some other party operating directly on ciphertexts,
  * while peers still use the same envelope protocol.
