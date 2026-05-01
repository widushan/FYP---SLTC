import os
import cv2
import numpy as np
import json
import base64
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D,
                                     GlobalAveragePooling2D, GlobalAveragePooling1D,
                                     TimeDistributed, Dense, Dropout, Layer,
                                     concatenate, MultiHeadAttention)
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__, template_folder=os.path.dirname(__file__))
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

# ─── CONFIG ──────────────────────────────────────────────
IMG_SIZE    = (64, 64)
GLOBAL_SIZE = (128, 128)
FEATURES = [
    "AU01","AU02","AU04","AU05","AU06","AU07","AU45",
    "AU09","AU10","AU12","AU14","AU15","AU20","AU23",
    "AU24","AU25","AU26","AU17"
]
SEQUENCE_LENGTH = len(FEATURES)

# Landmark-based ROI Map (MediaPipe 468 → approximate AU regions)
AU_MEDIAPIPE_MAP = {
    "AU01": [107, 336],           # Inner brow
    "AU02": [46, 276],            # Outer brow
    "AU04": [107, 336],           # Brow lowerer
    "AU05": [159, 386],           # Upper lid
    "AU06": [116, 345],           # Cheek raiser
    "AU07": [159, 386],           # Lid tightener
    "AU09": [98, 327],            # Nose wrinkler
    "AU10": [61, 291],            # Upper lip raiser
    "AU12": [61, 291],            # Lip corner puller
    "AU14": [61, 291],            # Dimpler
    "AU15": [17, 314],            # Lip corner depressor
    "AU20": [61, 291],            # Lip stretcher
    "AU23": [0, 17],              # Lip tightener
    "AU24": [0, 17],              # Lip pressor
    "AU25": [13, 14],             # Lips part
    "AU26": [152, 175],           # Jaw drop
    "AU17": [152, 199],           # Chin raiser
    "AU45": [159, 386],           # Blink
}

APP_ROOT = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(APP_ROOT, 'weights')
if not os.path.isdir(WEIGHTS_DIR):
    WEIGHTS_DIR = APP_ROOT

TEXT_EMB_DIR = os.path.join(APP_ROOT, 'embeddings')
if not os.path.isdir(TEXT_EMB_DIR):
    TEXT_EMB_DIR = APP_ROOT

model_cache = {}
emb_healthy = None
emb_parkinson = None

# ─── MODEL LAYERS ────────────────────────────────────────
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight("att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight("att_bias",   shape=(input_shape[1],  1), initializer="zeros")
        super().build(input_shape)
    def call(self, x):
        et = tf.squeeze(tf.tanh(tf.matmul(x, self.W) + self.b), axis=-1)
        return tf.reduce_sum(x * tf.expand_dims(tf.nn.softmax(et), -1), axis=1)
    def get_config(self): return super().get_config()

class DynamicGraphConv(Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
    def build(self, input_shape):
        self.W = self.add_weight("W", shape=(input_shape[-1], self.channels), initializer="normal")
        super().build(input_shape)
    def call(self, U):
        U_norm = tf.math.l2_normalize(U, axis=-1)
        A = tf.nn.softmax(tf.matmul(U_norm, U_norm, transpose_b=True) * 5.0, axis=-1)
        # Using tf.matmul with A and (U @ W)
        return tf.nn.relu(U + tf.matmul(A, tf.matmul(U, self.W)))
    def get_config(self):
        cfg = super().get_config(); cfg["channels"] = self.channels; return cfg

class DDCA(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
    def build(self, input_shape):
        self.mha_u_z = MultiHeadAttention(num_heads=4, key_dim=self.dim)
        self.mha_z_u = MultiHeadAttention(num_heads=4, key_dim=self.dim)
        super().build(input_shape)
    def call(self, inputs):
        U, Z    = inputs
        U_prime = self.mha_u_z(query=U, value=Z, key=Z)
        Z_prime = self.mha_z_u(query=Z, value=U, key=U)
        return U_prime + Z_prime
    def get_config(self):
        cfg = super().get_config(); cfg["dim"] = self.dim; return cfg

class CDCA(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
    def build(self, input_shape):
        self.mha = MultiHeadAttention(num_heads=4, key_dim=self.dim)
        super().build(input_shape)
    def call(self, inputs):
        G_expanded, Z = inputs
        return self.mha(query=G_expanded, value=Z, key=Z)
    def get_config(self):
        cfg = super().get_config(); cfg["dim"] = self.dim; return cfg

# ─── MODEL BUILDER ───────────────────────────────────────
def build_cnn(inp_shape=(64, 64, 3)):
    inp = Input(shape=inp_shape)
    x = inp
    for filters in [16, 32, 64]:
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    return Model(inputs=inp, outputs=x)

def build_hybrid_model(emb_healthy, emb_parkinson):
    # ── Inputs ──────────────────────────────────────────────────
    loc_in  = Input(shape=(SEQUENCE_LENGTH, 64, 64, 3), name="local_input")
    glob_in = Input(shape=(128, 128, 3),                name="global_input")

    # ── Local Visual Encoder ─────────────────────────────────────
    U = TimeDistributed(build_cnn(inp_shape=(64, 64, 3)))(loc_in)  # (B, 18, 64)
    U = BatchNormalization()(U)

    # ── Global Visual Encoder ────────────────────────────────────
    G          = build_cnn(inp_shape=(128, 128, 3))(glob_in)           # (B, 64)
    G_expanded = tf.keras.layers.RepeatVector(SEQUENCE_LENGTH)(G)       # (B, 18, 64)

    # ── Text Embedding Injection ─────────────────────────────────
    def get_emb(emb_array):
        def fn(u_tensor):
            emb    = tf.constant(emb_array, dtype=tf.float32)
            b_size = tf.shape(u_tensor)[0]
            return tf.tile(tf.expand_dims(emb, 0), [b_size, 1, 1])
        return fn

    Z_h  = tf.keras.layers.Lambda(get_emb(emb_healthy),   name="emb_healthy")(U)    # (B, 18, 512)
    Z_pd = tf.keras.layers.Lambda(get_emb(emb_parkinson), name="emb_parkinson")(U)  # (B, 18, 512)

    # ── Text Projection (Matching H5 shapes: 512 -> 64) ───────────
    Z_h_proj  = Dense(64, activation='relu', name="text_proj_healthy")(Z_h)    # (B, 18, 64)
    Z_pd_proj = Dense(64, activation='relu', name="text_proj_parkinson")(Z_pd)  # (B, 18, 64)
    
    # ── Graph Reasoning (visual-only) ────────────────────────────
    # Note: DynamicGraphConv now uses raw weight tensor 'W' to match H5
    U_graph = DynamicGraphConv(64)(U)   # (B, 18, 64)
    
    # ── Vision-Language Interaction ──────────────────────────────
    # DDCA: local vision ↔ parkinson text
    D = DDCA(64)([U, Z_pd_proj])
    
    # CDCA: global face context → healthy text
    C = CDCA(64)([G_expanded, Z_h_proj])
    
    # ── Fusion ───────────────────────────────────────────────────
    interact_fusion = Dense(64, activation='relu')(concatenate([D, C], axis=-1))

    x_vision = concatenate([
        GlobalAveragePooling1D()(interact_fusion),   # (B, 64) — vision-language
        GlobalAveragePooling1D()(U_graph),            # (B, 64) — visual spatial GCN
        G                                             # (B, 64) — global face
    ])
    x_vision = BatchNormalization()(x_vision)
    x_vision = Dense(64, activation='relu')(x_vision)

    # ── Classification Head ──────────────────────────────────────
    x   = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x_vision)
    x   = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[loc_in, glob_in], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_embeddings():
    global emb_healthy, emb_parkinson
    try:
        emb_healthy   = np.load(os.path.join(TEXT_EMB_DIR, 'text_embeddings_healthy.npy')).astype(np.float32)
        emb_parkinson = np.load(os.path.join(TEXT_EMB_DIR, 'text_embeddings_parkinson.npy')).astype(np.float32)
        print(f"[EMB] Loaded dual embeddings: {emb_healthy.shape}, {emb_parkinson.shape}")
    except FileNotFoundError:
        try:
            combined = np.load(os.path.join(TEXT_EMB_DIR, 'text_embeddings.npy')).astype(np.float32)
            emb_healthy = emb_parkinson = combined
            print("[EMB] Using combined embeddings (v2 fallback)")
        except FileNotFoundError:
            emb_healthy = emb_parkinson = np.zeros((18, 512), dtype=np.float32)
            print("[EMB] WARNING: No embeddings found, using zeros")

def load_ensemble_models():
    global model_cache
    if not os.path.isdir(WEIGHTS_DIR):
        print(f"[WARN] No weights directory found at {WEIGHTS_DIR}")
        return
    
    weight_files = sorted([f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.weights.h5')])
    if not weight_files:
        print("[WARN] No .weights.h5 files found")
        return
    
    import h5py
    import re
    def manual_load(model, filepath):
        print(f"  [LOAD] Loading {os.path.basename(filepath)}")
        f = h5py.File(filepath, 'r')
        def assign_weights(layer, h5_group, depth=0):
            indent = "  " * depth
            
            # Sub-layers are either direct keys or in a 'layers' subgroup
            target_group = h5_group['layers'] if 'layers' in h5_group else h5_group
            
            if hasattr(layer, 'layers'):
                type_counters = {}
                for sub_layer in layer.layers:
                    base_name = re.sub(r'_\d+$', '', sub_layer.name)
                    if sub_layer.name not in target_group and base_name not in target_group:
                        base_name = sub_layer.__class__.__name__.lower()
                        base_name = re.sub(r'_\d+$', '', base_name)
                    
                    count = type_counters.get(base_name, 0)
                    type_counters[base_name] = count + 1
                    expected_key = base_name if count == 0 else f"{base_name}_{count}"
                    
                    if expected_key in target_group:
                        assign_weights(sub_layer, target_group[expected_key], depth + 1)
                    elif sub_layer.name in target_group:
                        assign_weights(sub_layer, target_group[sub_layer.name], depth + 1)
                
            if 'layer' in h5_group and hasattr(layer, 'layer'):
                assign_weights(layer.layer, h5_group['layer'], depth + 1)
                
            for k in h5_group.keys():
                if k in ['layers', 'layer', 'vars']: continue
                if hasattr(layer, k):
                    assign_weights(getattr(layer, k), h5_group[k], depth + 1)
                elif hasattr(layer, f"_{k}"):
                    assign_weights(getattr(layer, f"_{k}"), h5_group[k], depth + 1)
                
            if 'vars' in h5_group:
                vars_group = h5_group['vars']
                weights = []
                for str_i in map(str, range(len(vars_group))):
                    if str_i in vars_group:
                        weights.append(vars_group[str_i][()])
                if weights:
                    try:
                        if hasattr(layer, 'set_weights'):
                            layer.set_weights(weights)
                            name = getattr(layer, 'name', 'unknown')
                        else:
                            # Direct variable assignment
                            layer.assign(weights[0])
                            name = "weight_tensor"
                        print(f"{indent}[OK] Loaded weights for {name}")
                    except Exception as e:
                        name = getattr(layer, 'name', 'variable')
                        print(f"{indent}[FAIL] Failed weights for {name}: {e}")
        
        assign_weights(model, f['layers'])
        f.close()

    print(f"[MODEL] Loading {len(weight_files)} ensemble models...")
    for wf in weight_files:
        try:
            tf.keras.backend.clear_session()
            m = build_hybrid_model(emb_healthy, emb_parkinson)
            manual_load(m, os.path.join(WEIGHTS_DIR, wf))
            model_cache[wf] = m
            print(f"  [OK] {wf}")
        except Exception as e:
            print(f"  [FAIL] {wf}: {e}")
    print(f"[MODEL] {len(model_cache)} models loaded.")

# ─── FACE PROCESSING ─────────────────────────────────────
def extract_face_frame(video_path):
    """Extract best face frame from video using MediaPipe.
    
    'Best' means: sharpest focus + most frontal face (landmarks spread wide).
    This matters especially for short webcam WebM blobs where:
      - total_frames is 0 (no index in MediaRecorder WebM)
      - any blurry/angled frame can corrupt the AU patches
    """
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"[DEBUG] extract_face_frame: video={video_path}, total_frames={total_frames}, fps={fps:.1f}")

    # For short webcam WebM blobs, total_frames is often 0 or wrong.
    # Use sequential scan for everything ≤ 10s or when frame count is unreliable.
    is_short_or_unreliable = (total_frames <= 0) or (total_frames / max(fps, 1) < 10)

    best_frame     = None
    best_landmarks = None
    best_score     = -1.0   # higher = sharper + more frontal

    def frame_score(frame, landmarks):
        """Score = Laplacian sharpness * face_width_ratio."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Face width in normalized coords (cheek to cheek: lm 234 vs 454)
        lm = landmarks
        face_width = abs(lm[454].x - lm[234].x)
        return sharpness * (face_width + 0.1)

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:

        if not is_short_or_unreliable:
            # Long uploaded video: sample from the middle third
            start = total_frames // 3
            end   = 2 * total_frames // 3
            step  = max(1, (end - start) // 30)

            for fi in range(start, end, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    sc = frame_score(frame, lm)
                    if sc > best_score:
                        best_score     = sc
                        best_frame     = frame
                        best_landmarks = lm
                        print(f"[DEBUG] Better frame at {fi}, score={sc:.1f}")

        # Sequential scan: covers webcam blobs AND acts as fallback for long videos
        if best_frame is None or is_short_or_unreliable:
            print("[DEBUG] Sequential scan (short/unreliable or fallback).")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0
            # No warmup skip for short recordings — every frame counts.
            # For longer files keep a small skip (0.5s) to avoid the very first frame.
            warmup = 0 if is_short_or_unreliable else int(fps * 0.5)
            max_scan = int(fps * 20)   # scan up to 20s max

            while count < max_scan:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                count += 1
                if count <= warmup:
                    continue
                if count % 3 != 0:   # sample every 3rd frame for speed
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    sc = frame_score(frame, lm)
                    if sc > best_score:
                        best_score     = sc
                        best_frame     = frame
                        best_landmarks = lm
                        print(f"[DEBUG] Better frame at count={count}, score={sc:.1f}")

    cap.release()
    if best_frame is None:
        print("[DEBUG] No face found in video.")
    else:
        print(f"[DEBUG] Best frame selected, score={best_score:.1f}")
    return best_frame, best_landmarks


def crop_au_patches(frame, landmarks):
    """Crop AU patches + global face from frame."""
    h, w = frame.shape[:2]
    lm = landmarks

    def lm_xy(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    # Global face bbox
    # Match OpenFace 68 landmark bounding box (eyebrows to chin, cheek to cheek)
    used_indices = set()
    for indices in AU_MEDIAPIPE_MAP.values():
        used_indices.update(indices)
    used_indices.update([234, 454, 152]) # Left cheek, right cheek, bottom chin
    
    xs = [int(lm[i].x * w) for i in used_indices]
    ys = [int(lm[i].y * h) for i in used_indices]
    pad = 20
    gx1, gy1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
    gx2, gy2 = min(w, max(xs)+pad), min(h, max(ys)+pad)
    global_crop = frame[gy1:gy2, gx1:gx2]
    if global_crop.size == 0:
        return None, None
    global_img = cv2.resize(global_crop, GLOBAL_SIZE)
    global_img = cv2.cvtColor(global_img, cv2.COLOR_BGR2RGB)

    patches = []
    for au in FEATURES:
        indices = AU_MEDIAPIPE_MAP.get(au, [0])
        pts = [lm_xy(i) for i in indices]
        cx  = int(np.mean([p[0] for p in pts]))
        cy  = int(np.mean([p[1] for p in pts]))
        r   = 32
        x1, y1 = max(0, cx-r), max(0, cy-r)
        x2, y2 = min(w, cx+r), min(h, cy+r)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            patch = np.zeros((64, 64, 3), dtype=np.uint8)
        patch = cv2.resize(patch, IMG_SIZE)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patches.append(patch)

    return np.array(patches, dtype=np.float32) / 255.0, \
           np.array(global_img, dtype=np.float32) / 255.0


def run_inference(local_patches, global_img):
    """Run ensemble inference and return probability + confidence."""
    if not model_cache:
        # Demo mode — return mock result
        np.random.seed(42)
        prob = float(np.random.uniform(0.3, 0.8))
        return prob, 0.72, True

    X_loc  = local_patches[np.newaxis]   # (1, 18, 64, 64, 3)
    X_glob = global_img[np.newaxis]      # (1, 128, 128, 3)

    probs = []
    for model in model_cache.values():
        p = float(model.predict([X_loc, X_glob], verbose=0).flatten()[0])
        probs.append(p)

    mean_prob   = float(np.mean(probs))
    std_prob    = float(np.std(probs))
    confidence  = float(1.0 - 2 * std_prob)  # higher agreement → higher confidence
    confidence  = max(0.0, min(1.0, confidence))
    
    print(f"[INFERENCE] Result: {mean_prob:.4f} (conf: {confidence:.2f}) from {len(probs)} models")
    return mean_prob, confidence, False


# ─── ROUTES ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        'models_loaded': len(model_cache),
        'embeddings_loaded': emb_healthy is not None,
        'demo_mode': len(model_cache) == 0
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    tmp_path = None
    mp4_path = None
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        content_type = video_file.content_type or ''
        is_webm = 'webm' in content_type or 'ogg' in content_type

        # Save the raw upload
        suffix = '.mp4' if 'mp4' in content_type else '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        print(f"[ANALYZE] Saved upload: {tmp_path}, content_type={content_type}, is_webm={is_webm}")

        # WebM from MediaRecorder is often poorly seekable and OpenCV decodes
        # it incorrectly on many platforms → transcode to MP4 first.
        process_path = tmp_path
        if is_webm:
            mp4_path = tmp_path.replace('.webm', '_converted.mp4')
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y',
                '-i', tmp_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '18',          # high quality — important for AU patches
                '-an',                 # no audio needed
                '-movflags', '+faststart',
                mp4_path
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                print(f"[ANALYZE] ffmpeg transcode OK → {mp4_path}")
                process_path = mp4_path
            else:
                print(f"[ANALYZE] ffmpeg failed (rc={result.returncode}), using raw WebM")
                print(f"[ANALYZE] ffmpeg stderr: {result.stderr[-500:]}")
                # Fall through with original — best effort

        # Extract face
        frame, landmarks = extract_face_frame(process_path)
        if frame is None or landmarks is None:
            return jsonify({'error': 'No face detected in video. Please ensure your face is clearly visible and well-lit.'}), 422

        # Crop patches
        local_patches, global_img = crop_au_patches(frame, landmarks)
        if local_patches is None:
            return jsonify({'error': 'Failed to extract facial patches.'}), 422

        # Inference
        prob, confidence, demo = run_inference(local_patches, global_img)

        label      = 'Parkinson' if prob > 0.5 else 'Healthy'
        confidence_pct = round(confidence * 100, 1)
        prob_pct       = round(prob * 100, 1)

        # Encode the analyzed face frame for display
        _, jpg = cv2.imencode('.jpg', frame)
        face_b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')

        return jsonify({
            'label':       label,
            'probability': prob_pct,
            'confidence':  confidence_pct,
            'demo_mode':   demo,
            'face_image':  face_b64,
            'models_used': len(model_cache)
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if mp4_path and os.path.exists(mp4_path):
            os.unlink(mp4_path)


if __name__ == '__main__':
    load_embeddings()
    load_ensemble_models()
    app.run(debug=False, host='0.0.0.0', port=5050)