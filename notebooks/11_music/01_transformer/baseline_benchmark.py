"""Re-trains the ORIGINAL (baseline) architecture from transformer.ipynb on the same
train/validation split used by transformer-modified.ipynb, then measures held-out
loss/accuracy, generation quality and generation latency with the ORIGINAL sampler.
Writes output/baseline_metrics.json and output/baseline_sample.mid."""
import os, sys, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
import music21

here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(here)
from transformer_utils import (load_parsed_files, get_midi_note, SinePositionEncoding,
                               split_sequences, tokenize_sequences, encode_with,
                               generation_metrics)

tf.keras.utils.set_random_seed(42)
OUT = os.path.join(here, "output"); os.makedirs(OUT, exist_ok=True)

# ---- hyper-parameters exactly as in transformer.ipynb --------------------------------
SEQ_LEN, EMBEDDING_DIM, KEY_DIM, N_HEADS = 50, 256, 256, 5
DROPOUT_RATE, FEED_FORWARD_DIM, EPOCHS, BATCH_SIZE = 0.3, 256, 50, 256
N_SAMPLES, TEMPERATURE = 20, 0.5

notes, durations = load_parsed_files(os.path.join(here, "parsed_data"))
(tr_n, tr_d), (va_n, va_d) = split_sequences(notes, durations, SEQ_LEN + 1)
(nvec, notes_vocab), (dvec, durations_vocab) = tokenize_sequences(notes, durations)
tr_n, tr_d, va_n, va_d = [encode_with(v, x) for v, x in
                          ((nvec, tr_n), (dvec, tr_d), (nvec, va_n), (dvec, va_d))]
NV, DV = len(notes_vocab), len(durations_vocab)

def to_ds(n, d, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(((n[:, :-1], d[:, :-1]), (n[:, 1:], d[:, 1:])))
    if shuffle: ds = ds.shuffle(5000, seed=42)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_ds, val_ds = to_ds(tr_n, tr_d, True), to_ds(va_n, va_d, False)

# ---- original layers (verbatim logic) -------------------------------------------------
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]; j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.reshape(tf.cast(m, dtype), [1, n_dest, n_src])
    mult = tf.concat([tf.expand_dims(batch_size, -1), [1, 1]], axis=0)
    return tf.tile(mask, mult)

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, name="attention", dropout_rate=DROPOUT_RATE, **kw):
        super().__init__(name=name, **kw)
        self.attn = layers.MultiHeadAttention(num_heads, key_dim, output_shape=embed_dim)
        self.dropout_1 = layers.Dropout(dropout_rate); self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(ff_dim, activation="relu"); self.ffn_2 = layers.Dense(embed_dim)
        self.dropout_2 = layers.Dropout(dropout_rate); self.ln_2 = layers.LayerNormalization(epsilon=1e-6)
    def call(self, inputs):
        s = tf.shape(inputs)
        mask = causal_attention_mask(s[0], s[1], s[1], tf.bool)
        a, scores = self.attn(inputs, inputs, attention_mask=mask, return_attention_scores=True)
        out1 = self.ln_1(inputs + self.dropout_1(a))
        f = self.dropout_2(self.ffn_2(self.ffn_1(out1)))
        return self.ln_2(out1 + f), scores

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, **kw):
        super().__init__(**kw)
        self.token_emb = layers.Embedding(vocab_size, embed_dim, embeddings_initializer="he_uniform")
        self.pos_emb = SinePositionEncoding()
    def call(self, x):
        e = self.token_emb(x); return e + self.pos_emb(e)

ni = layers.Input(shape=(None,), dtype=tf.int32); di = layers.Input(shape=(None,), dtype=tf.int32)
emb = layers.Concatenate()([TokenAndPositionEmbedding(NV, EMBEDDING_DIM // 2)(ni),
                            TokenAndPositionEmbedding(DV, EMBEDDING_DIM // 2)(di)])
x, att_scores = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM)(emb)
no = layers.Dense(NV, activation="softmax", name="note_outputs")(x)
do = layers.Dense(DV, activation="softmax", name="duration_outputs")(x)
model = models.Model([ni, di], [no, do])
model.compile("adam", loss=[losses.SparseCategoricalCrossentropy()] * 2,
              metrics=[["accuracy"], ["accuracy"]])
att_model = models.Model([ni, di], att_scores)
print("params:", model.count_params())

t0 = time.time()
# same stopping rule as transformer-modified.ipynb so the comparison is fair
es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[es], verbose=2)
train_time = time.time() - t0
print("train time", train_time)

# ---- original sampler / generation loop (model.predict twice per token) ---------------
i2n, i2d = list(notes_vocab), list(durations_vocab)
n2i = {n: i for i, n in enumerate(i2n)}; d2i = {d: i for i, d in enumerate(i2d)}

def sample_from(p, T):
    p = p ** (1 / T); p = p / np.sum(p)
    return np.random.choice(len(p), p=p), p

def generate(start_notes, start_durs, max_tokens, T):
    sn = [n2i.get(x, 1) for x in start_notes]; sd = [d2i.get(x, 1) for x in start_durs]
    sn_s, sd_s = list(start_notes), list(start_durs)
    stream = music21.stream.Stream(); stream.append(music21.clef.BassClef())
    for a, b in zip(start_notes, start_durs):
        m = get_midi_note(a, b)
        if m is not None: stream.append(m)
    while len(sn) < max_tokens:
        x1, x2 = np.array([sn]), np.array([sd])
        pn, pd_ = model.predict([x1, x2], verbose=0)
        repeat = True
        while repeat:
            ni_ = 1
            while ni_ == 1: ni_, _ = sample_from(pn[0][-1], T)
            di_ = 1
            while di_ == 1: di_, _ = sample_from(pd_[0][-1], T)
            note, dur = i2n[ni_], i2d[di_]
            m = get_midi_note(note, dur)
            repeat = isinstance(m, (music21.chord.Chord, music21.note.Note, music21.note.Rest)) and dur == "0.0"
        if m is not None: stream.append(m)
        att_model.predict([x1, x2], verbose=0)   # the original also runs this every token
        sn.append(ni_); sd.append(di_); sn_s.append(note); sd_s.append(dur)
        if note == "START": break
    return stream, sn_s, sd_s

np.random.seed(42)
gen_n, gen_d, times = [], [], []
for k in range(N_SAMPLES):
    t0 = time.time()
    stream, sn_s, sd_s = generate(["START"], ["0.0"], 50, TEMPERATURE)
    times.append((time.time() - t0) / max(len(sn_s) - 1, 1))
    gen_n.append(sn_s[1:]); gen_d.append(sd_s[1:])
    if k == 0:
        stream.chordify().write("midi", fp=os.path.join(OUT, "baseline_sample.mid"))
        first = {"notes": sn_s, "durations": sd_s}

ev = model.evaluate(val_ds, verbose=0, return_dict=True)
res = {
    "params": int(model.count_params()), "epochs": len(hist.history["loss"]), "train_time_s": train_time,
    "train_last": {k: float(v[-1]) for k, v in hist.history.items()},
    "val_eval": {k: float(v) for k, v in ev.items()},
    "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
    "generation": {"temperature": TEMPERATURE, "n_samples": N_SAMPLES,
                   "ms_per_token_mean": 1000 * float(np.mean(times)),
                   "ms_per_token_std": 1000 * float(np.std(times)),
                   "metrics": generation_metrics(gen_n, gen_d), "first_sample": first},
}
# reference statistics: real Bach windows (first 49 tokens after START-free windows)
idx = np.random.RandomState(0).choice(len(va_n), N_SAMPLES, replace=False)
res["real_bach_reference"] = generation_metrics(
    [[i2n[t] for t in va_n[i][:50]] for i in idx], [[i2d[t] for t in va_d[i][:50]] for i in idx])
json.dump(res, open(os.path.join(OUT, "baseline_metrics.json"), "w"), indent=2)
print(json.dumps({k: res[k] for k in ("val_eval", "generation", "real_bach_reference")}, indent=1)[:3000])
