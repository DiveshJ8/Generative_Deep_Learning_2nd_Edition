import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from keras import layers, models, losses, callbacks
import music21
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = r"d:\Sem3\Generative_Deep_Learning_2nd_Edition"
transformer_dir = os.path.join(repo_root, "notebooks", "11_music", "01_transformer")
sys.path.append(transformer_dir)

from transformer_utils import load_parsed_files, get_midi_note, SinePositionEncoding

print("=== Starting Modernized Music Transformer Training & Evaluation ===")

# Parameters
PARSED_DATA_PATH = os.path.join(transformer_dir, "parsed_data")
OUTPUT_DIR = os.path.join(transformer_dir, "output")
MODELS_DIR = os.path.join(transformer_dir, "models")
CHECKPOINT_DIR = os.path.join(transformer_dir, "checkpoint")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEQ_LEN = 50
EMBEDDING_DIM = 128
KEY_DIM = 64
N_HEADS = 4
DROPOUT_RATE = 0.2
FEED_FORWARD_DIM = 256
BATCH_SIZE = 256
EPOCHS = 8
GENERATE_LEN = 50

# 1. Load data
notes, durations = load_parsed_files(PARSED_DATA_PATH)
print(f"Loaded {len(notes)} note-duration sequence pairs.")

# 2. Vectorize
notes_vec = layers.TextVectorization(standardize=None, output_mode="int")
notes_vec.adapt(tf.data.Dataset.from_tensor_slices(notes).batch(BATCH_SIZE))
dur_vec = layers.TextVectorization(standardize=None, output_mode="int")
dur_vec.adapt(tf.data.Dataset.from_tensor_slices(durations).batch(BATCH_SIZE))

notes_vocab = notes_vec.get_vocabulary()
durations_vocab = dur_vec.get_vocabulary()
notes_vocab_size = len(notes_vocab)
durations_vocab_size = len(durations_vocab)

print(f"Notes Vocab Size: {notes_vocab_size}")
print(f"Durations Vocab Size: {durations_vocab_size}")

note_to_index = {note: i for i, note in enumerate(notes_vocab)}
index_to_note = {i: note for i, note in enumerate(notes_vocab)}
duration_to_index = {dur: i for i, dur in enumerate(durations_vocab)}
index_to_duration = {i: dur for i, dur in enumerate(durations_vocab)}

# 3. Create Dataset Pipeline
def prepare_inputs(n, d):
    n_tok = notes_vec(tf.expand_dims(n, -1))
    d_tok = dur_vec(tf.expand_dims(d, -1))
    x = (n_tok[:, :-1], d_tok[:, :-1])
    y = (n_tok[:, 1:], d_tok[:, 1:])
    return x, y

raw_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(notes),
    tf.data.Dataset.from_tensor_slices(durations)
))

train_ds = (
    raw_ds.shuffle(buffer_size=5000, seed=42)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# 4. Architecture
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat([tf.expand_dims(batch_size, -1), [1, 1]], axis=0)
    return tf.tile(mask, mult)

@tf.keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            embeddings_initializer="he_uniform",
        )
        self.pos_emb = SinePositionEncoding()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        emb = self.token_emb(x)
        pos = self.pos_emb(emb)
        return self.dropout(emb + pos)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "dropout_rate": self.dropout_rate,
        })
        return cfg

@tf.keras.utils.register_keras_serializable()
class PreLNTransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(num_heads, key_dim, output_shape=embed_dim)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(ff_dim, activation="gelu")
        self.ffn_2 = layers.Dense(embed_dim)
        self.dropout_2 = layers.Dropout(dropout_rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        bs = input_shape[0]
        sl = input_shape[1]
        mask = causal_attention_mask(bs, sl, sl, tf.bool)
        # Pre-LN branch 1: Multi-head attention
        norm_1 = self.ln_1(inputs)
        attn_out, attn_scores = self.attn(
            norm_1, norm_1, attention_mask=mask, return_attention_scores=True
        )
        x = inputs + self.dropout_1(attn_out)
        # Pre-LN branch 2: GELU Feed-Forward Network
        norm_2 = self.ln_2(x)
        ffn_out = self.dropout_2(self.ffn_2(self.ffn_1(norm_2)))
        return x + ffn_out, attn_scores

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return cfg

# Build Model with Harmonic-Temporal Fusion and Stacked Transformer Blocks
note_inputs = layers.Input(shape=(None,), dtype=tf.int32, name="note_inputs")
dur_inputs = layers.Input(shape=(None,), dtype=tf.int32, name="durations_inputs")

note_emb = TokenAndPositionEmbedding(
    notes_vocab_size, EMBEDDING_DIM // 2, dropout_rate=DROPOUT_RATE, name="note_emb"
)(note_inputs)
dur_emb = TokenAndPositionEmbedding(
    durations_vocab_size, EMBEDDING_DIM // 2, dropout_rate=DROPOUT_RATE, name="dur_emb"
)(dur_inputs)

concat_emb = layers.Concatenate(name="concat_emb")([note_emb, dur_emb])
# Harmonic-temporal fusion dense projection
fused_emb = layers.Dense(EMBEDDING_DIM, activation="gelu", name="harmonic_temporal_fusion")(concat_emb)
fused_emb = layers.LayerNormalization(epsilon=1e-6, name="fusion_ln")(fused_emb)

# Deepened 2-layer Transformer Backbone
x1, att1 = PreLNTransformerBlock(
    N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, dropout_rate=DROPOUT_RATE, name="transformer_block_1"
)(fused_emb)
x2, att2 = PreLNTransformerBlock(
    N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, dropout_rate=DROPOUT_RATE, name="transformer_block_2"
)(x1)

note_outputs = layers.Dense(notes_vocab_size, activation="softmax", name="note_outputs")(x2)
dur_outputs = layers.Dense(durations_vocab_size, activation="softmax", name="duration_outputs")(x2)

model = models.Model(
    inputs=[note_inputs, dur_inputs],
    outputs=[note_outputs, dur_outputs],
    name="ModernizedMusicTransformer"
)

infer_model = models.Model(
    inputs=[note_inputs, dur_inputs],
    outputs=[note_outputs, dur_outputs, att2],
    name="InferenceMusicTransformer"
)

optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
model.compile(
    optimizer=optimizer,
    loss=[
        losses.SparseCategoricalCrossentropy(),
        losses.SparseCategoricalCrossentropy(),
    ],
    metrics=[["accuracy"], ["accuracy"]]
)

model.summary()

# 5. Fast Direct Graph Music Generator
class FastMusicGenerator(callbacks.Callback):
    def __init__(self, index_to_note, index_to_duration, infer_model=None, top_k=20, top_p=0.90, repetition_penalty=1.15):
        super().__init__()
        self.index_to_note = index_to_note
        self.index_to_duration = index_to_duration
        self.infer_model = infer_model
        self.note_to_index = {v: k for k, v in index_to_note.items()}
        self.duration_to_index = {v: k for k, v in index_to_duration.items()}
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def sample_from_probs(self, probs, temperature, recent_tokens=None):
        probs = np.array(probs, dtype=np.float64)
        if temperature <= 0.05:
            return int(np.argmax(probs)), probs

        # Convert back to pseudo-logits for stabilized temperature and penalties
        eps = 1e-12
        logits = np.log(np.clip(probs, eps, 1.0))

        if recent_tokens and self.repetition_penalty > 1.0:
            for tok in set(recent_tokens[-12:]):
                if tok < len(logits):
                    if logits[tok] < 0:
                        logits[tok] *= self.repetition_penalty
                    else:
                        logits[tok] /= self.repetition_penalty

        scaled_logits = logits / temperature

        if self.top_k > 0 and self.top_k < len(scaled_logits):
            indices_to_remove = np.argsort(scaled_logits)[:-self.top_k]
            scaled_logits[indices_to_remove] = -1e9

        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        filtered_probs = exp_logits / np.sum(exp_logits)

        if self.top_p < 1.0:
            sorted_indices = np.argsort(filtered_probs)[::-1]
            sorted_probs = filtered_probs[sorted_indices]
            cum_probs = np.cumsum(sorted_probs)
            remove_mask = cum_probs > self.top_p
            remove_mask[1:] = remove_mask[:-1]
            remove_mask[0] = False
            filtered_probs[sorted_indices[remove_mask]] = 0.0
            filtered_probs = filtered_probs / np.sum(filtered_probs)

        sampled_idx = int(np.random.choice(len(filtered_probs), p=filtered_probs))
        return sampled_idx, filtered_probs

    def get_note(self, note_probs, dur_probs, temperature, recent_notes):
        sample_note_idx = 1
        attempts = 0
        while sample_note_idx == 1 and attempts < 10:
            sample_note_idx, filtered_n_probs = self.sample_from_probs(
                note_probs, temperature, recent_notes
            )
            sample_note = self.index_to_note.get(sample_note_idx, "rest")
            attempts += 1

        sample_dur_idx = 1
        attempts = 0
        while sample_dur_idx == 1 and attempts < 10:
            sample_dur_idx, filtered_d_probs = self.sample_from_probs(
                dur_probs, temperature
            )
            sample_dur = self.index_to_duration.get(sample_dur_idx, "0.25")
            attempts += 1

        new_note = get_midi_note(sample_note, sample_dur)
        return new_note, sample_note_idx, sample_note, filtered_n_probs, sample_dur_idx, sample_dur, filtered_d_probs

    def generate(self, start_notes, start_durations, max_tokens=50, temperature=0.7):
        start_note_tokens = [self.note_to_index.get(x, 1) for x in start_notes]
        start_duration_tokens = [self.duration_to_index.get(x, 1) for x in start_durations]
        info = []
        midi_stream = music21.stream.Stream()
        midi_stream.append(music21.clef.BassClef())

        for s_n, s_d in zip(start_notes, start_durations):
            n = get_midi_note(s_n, s_d)
            if n is not None:
                midi_stream.append(n)

        while len(start_note_tokens) < max_tokens:
            x1 = tf.constant([start_note_tokens], dtype=tf.int32)
            x2 = tf.constant([start_duration_tokens], dtype=tf.int32)
            
            # Direct tensor graph execution
            inf = self.infer_model if self.infer_model is not None else self.model
            preds = inf([x1, x2], training=False)
            note_dist = preds[0][0, -1].numpy()
            dur_dist = preds[1][0, -1].numpy()
            att_mat = preds[2][0, :, -1, :].numpy()

            repeat = True
            loop_guard = 0
            while repeat and loop_guard < 10:
                loop_guard += 1
                (
                    new_note,
                    s_note_idx,
                    s_note,
                    n_probs,
                    s_dur_idx,
                    s_dur,
                    d_probs
                ) = self.get_note(note_dist, dur_dist, temperature, start_note_tokens)

                if (
                    isinstance(new_note, (music21.chord.Chord, music21.note.Note, music21.note.Rest))
                    and s_dur == "0.0"
                ):
                    repeat = True
                else:
                    repeat = False

            if new_note is not None:
                midi_stream.append(new_note)

            info.append({
                "prompt": [start_notes.copy(), start_durations.copy()],
                "midi": midi_stream,
                "chosen_note": (s_note, s_dur),
                "note_probs": n_probs,
                "duration_probs": d_probs,
                "atts": att_mat,
            })

            start_note_tokens.append(s_note_idx)
            start_duration_tokens.append(s_dur_idx)
            start_notes.append(s_note)
            start_durations.append(s_dur)

            if s_note == "START" and len(start_note_tokens) > 5:
                break

        return info

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            t0 = time.time()
            info = self.generate(["START"], ["0.0"], max_tokens=GENERATE_LEN, temperature=0.7)
            gen_time = time.time() - t0
            last_notes = [f"{n}:{d}" for n, d in zip(info[-1]["prompt"][0][-8:], info[-1]["prompt"][1][-8:])]
            print(f"Epoch {epoch+1} generation ({gen_time:.3f}s): ... {' '.join(last_notes)}")

fast_generator = FastMusicGenerator(index_to_note, index_to_duration, infer_model=infer_model)

# 6. Train the Model
print("\n--- Training Modernized Music Transformer ---")
start_train = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[fast_generator],
    verbose=1
)
train_time = time.time() - start_train
print(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes).")

# Save Weights and Checkpoints
weights_path = os.path.join(MODELS_DIR, "transformer_modified.weights.h5")
model.save_weights(weights_path)
print(f"Saved weights to {weights_path}")

# 7. Benchmarking Generation Latency: Baseline vs Modernized
print("\n--- Latency Benchmark: Direct Graph vs model.predict() ---")
test_notes = ["START", "G2", "D3", "B3"]
test_durs = ["0.0", "0.25", "0.25", "0.25"]
t_tokens1 = [note_to_index.get(x, 1) for x in test_notes]
t_tokens2 = [duration_to_index.get(x, 1) for x in test_durs]

# Measure direct graph
t0 = time.time()
for _ in range(50):
    x1 = tf.constant([t_tokens1], dtype=tf.int32)
    x2 = tf.constant([t_tokens2], dtype=tf.int32)
    preds = infer_model([x1, x2], training=False)
    p1 = preds[0][0, -1].numpy()
t_direct = (time.time() - t0) / 50

# Measure model.predict
t0 = time.time()
for _ in range(20):
    preds = model.predict([np.array([t_tokens1]), np.array([t_tokens2])], verbose=0)
t_predict = (time.time() - t0) / 20

print(f"Direct Graph Execution: {t_direct*1000:.2f} ms/token")
print(f"model.predict() Call:   {t_predict*1000:.2f} ms/token")
print(f"Speedup Factor:         {t_predict / t_direct:.2f}x")

# 8. Generation Experiments Across Seeds and Temperatures
print("\n--- Generating Music Across Temperatures and Seeds ---")
experiments = [
    {"name": "standard_seed_T03", "seed_notes": ["START"], "seed_durs": ["0.0"], "temp": 0.3},
    {"name": "standard_seed_T07", "seed_notes": ["START"], "seed_durs": ["0.0"], "temp": 0.7},
    {"name": "standard_seed_T12", "seed_notes": ["START"], "seed_durs": ["0.0"], "temp": 1.2},
    {"name": "suite1_prelude_seed_G2", "seed_notes": ["START", "G2", "D3", "B3"], "seed_durs": ["0.0", "0.25", "0.25", "0.25"], "temp": 0.7},
    {"name": "suite2_dminor_seed_D3", "seed_notes": ["START", "D3", "F3", "A3"], "seed_durs": ["0.0", "0.25", "0.25", "0.25"], "temp": 0.7},
]

results = {}
for exp in experiments:
    t0 = time.time()
    info = fast_generator.generate(
        exp["seed_notes"].copy(),
        exp["seed_durs"].copy(),
        max_tokens=GENERATE_LEN,
        temperature=exp["temp"]
    )
    gen_time = time.time() - t0
    midi_stream = info[-1]["midi"].chordify()
    
    midi_filename = f"generated_{exp['name']}.mid"
    midi_filepath = os.path.join(OUTPUT_DIR, midi_filename)
    midi_stream.write("midi", fp=midi_filepath)

    generated_notes = [note for note, dur in [step["chosen_note"] for step in info]]
    generated_durs = [dur for note, dur in [step["chosen_note"] for step in info]]

    # Parse text representations
    text_elements = []
    for el in (midi_stream.flatten() if hasattr(midi_stream, "flatten") else midi_stream.flat):
        text_elements.append(f"{{{float(el.offset):.2f}}} {el}")

    # Metrics
    unique_pitches = len(set(generated_notes))
    unique_durs = len(set(generated_durs))
    # 3-gram repetitions
    trigrams = [tuple(generated_notes[i:i+3]) for i in range(len(generated_notes)-2)]
    rep_trigrams = len(trigrams) - len(set(trigrams))
    rep_trigram_rate = rep_trigrams / max(len(trigrams), 1)

    results[exp["name"]] = {
        "temp": exp["temp"],
        "seed_notes": exp["seed_notes"],
        "seed_durs": exp["seed_durs"],
        "generation_time_s": gen_time,
        "token_count": len(generated_notes),
        "notes": generated_notes,
        "durations": generated_durs,
        "unique_pitches": unique_pitches,
        "unique_durs": unique_durs,
        "rep_trigram_rate": rep_trigram_rate,
        "text_score": text_elements[:25],
        "midi_file": midi_filename
    }
    print(f"Exp: {exp['name']} | Time: {gen_time:.3f}s | Unique Pitches: {unique_pitches} | Rep Trigram Rate: {rep_trigram_rate:.2%}")

# 9. Plotting Visualizations for the Best Sample (standard_seed_T07)
best_info = fast_generator.generate(["START"], ["0.0"], max_tokens=GENERATE_LEN, temperature=0.7)

# Pitch Distribution Grid
max_pitch = 70
seq_len = len(best_info)
grid = np.zeros((max_pitch, seq_len))
for j in range(seq_len):
    for i, prob in enumerate(best_info[j]["note_probs"]):
        try:
            pitch = music21.note.Note(index_to_note[i]).pitch.midi
            if pitch < max_pitch:
                grid[pitch, j] += prob
        except Exception:
            pass

plt.figure(figsize=(10, 8))
plt.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
plt.xlabel("Generation Step", fontsize=12)
plt.ylabel("MIDI Pitch", fontsize=12)
plt.title("Note Probability Trajectory (Modernized Transformer, T=0.7)", fontsize=14)
plt.colorbar(label="Probability")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "modified_note_probabilities.png"), dpi=200)
plt.close()

# Attention Plot
plot_size = min(20, len(best_info))
att_matrix = np.zeros((plot_size, plot_size))
for j in range(plot_size):
    atts = best_info[j]["atts"].max(axis=0) # max over heads
    att_matrix[j, :len(atts)] = atts[:plot_size]

plt.figure(figsize=(8, 8))
plt.imshow(att_matrix, cmap="magma", interpolation="nearest")
plt.xlabel("Key / Source Token Position", fontsize=12)
plt.ylabel("Query / Dest Token Position", fontsize=12)
plt.title("Causal Self-Attention Matrix (Layer 2, Top-Head)", fontsize=14)
plt.colorbar(label="Attention Weight")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "modified_attention_map.png"), dpi=200)
plt.close()

# Training history summary
training_summary = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_time_seconds": train_time,
    "final_loss": float(history.history.get("loss", [0])[-1]),
    "final_note_loss": float(history.history.get("note_outputs_loss", [0])[-1]),
    "final_dur_loss": float(history.history.get("duration_outputs_loss", [0])[-1]),
    "final_note_acc": float(history.history.get("note_outputs_accuracy", [0])[-1]),
    "final_dur_acc": float(history.history.get("duration_outputs_accuracy", [0])[-1]),
    "direct_graph_latency_ms": t_direct * 1000,
    "predict_call_latency_ms": t_predict * 1000,
    "speedup_factor": t_predict / t_direct,
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
}

with open(os.path.join(OUTPUT_DIR, "training_and_generation_metrics.json"), "w") as f:
    json.dump({"summary": training_summary, "experiments": results}, f, indent=2)

print("\n=== All Evaluation and Music Generation Complete! ===")
print("Summary metrics saved to training_and_generation_metrics.json")
