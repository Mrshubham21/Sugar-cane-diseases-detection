from flask import Flask, request, render_template, redirect, url_for, flash
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import base64
import sqlite3

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"

# =====================================
# CLASS NAMES (MUST MATCH TRAINING)
# =====================================

CLASS_NAMES = [
    "Dried",
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow"
]

# =====================================
# DISEASE INFO
# =====================================

DISEASE_INFO = {

    "Dried": {
        "treatment": "Improve irrigation immediately.",
        "prevention": "Avoid water stress.",
        "fertilizer": "Use moisture-retaining fertilizer."
    },

    "Healthy": {
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper irrigation and sunlight.",
        "fertilizer": "Use balanced NPK fertilizer."
    },

    "Mosaic": {
        "treatment": "Remove infected plants.",
        "prevention": "Use virus-free seeds and control aphids.",
        "fertilizer": "Apply micronutrient spray."
    },

    "RedRot": {
        "treatment": "Remove infected plants immediately.",
        "prevention": "Use resistant varieties.",
        "fertilizer": "Apply potassium-rich fertilizer."
    },

    "Rust": {
        "treatment": "Apply fungicide spray.",
        "prevention": "Ensure proper spacing between plants.",
        "fertilizer": "Use nitrogen control fertilizer."
    },

    "Yellow": {
        "treatment": "Improve soil condition.",
        "prevention": "Ensure proper drainage.",
        "fertilizer": "Use nitrogen-rich fertilizer."
    }
}

# =====================================
# CONFIG
# =====================================

MODEL_PATH = os.path.join(
    'model',
    'model_optimized.tflite'
)

IMG_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    'png',
    'jpg',
    'jpeg',
    'webp'
}

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# =====================================
# DATABASE
# =====================================

def init_db():

    conn = sqlite3.connect('data.db')

    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class TEXT,
            confidence REAL
        )
    ''')

    conn.commit()
    conn.close()

init_db()

# =====================================
# LOAD MODEL
# =====================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Loaded model:", MODEL_PATH)

# =====================================
# HELPERS
# =====================================

def allowed_file(filename):

    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_tflite_from_bytes(image_bytes):

    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert('RGB')

    img = img.resize(IMG_SIZE)

    # IMPORTANT:
    # NO /255 because model already rescales

    arr = np.array(img).astype(np.float32)

    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(
        input_details[0]['index'],
        arr
    )

    interpreter.invoke()

    out = interpreter.get_tensor(
        output_details[0]['index']
    )[0]

    # IMPORTANT:
    # model already uses softmax

    probs = out.astype(np.float32)

    idx = int(np.argmax(probs))

    confidence = float(probs[idx])

    # DEBUG
    print("\n========== Prediction ==========")

    for i, p in enumerate(probs):
        print(
            CLASS_NAMES[i],
            ":",
            round(float(p) * 100, 2),
            "%"
        )

    print("Predicted:", CLASS_NAMES[idx])
    print("Confidence:", confidence)

    print("================================\n")

    return CLASS_NAMES[idx], confidence

# =====================================
# HOME
# =====================================

@app.route('/')
def index():

    return render_template('upload.html')

# =====================================
# PREDICT ROUTE
# =====================================

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:

        flash("No file uploaded.")

        return redirect(url_for('index'))

    f = request.files['image']

    if f.filename == '':

        flash("No file selected.")

        return redirect(url_for('index'))

    if not allowed_file(f.filename):

        flash("Only JPG, PNG, JPEG, WEBP allowed.")

        return redirect(url_for('index'))

    # file size check

    f.seek(0, os.SEEK_END)

    size = f.tell()

    f.seek(0)

    if size == 0:

        flash("Empty file.")

        return redirect(url_for('index'))

    # validate image

    try:

        raw = f.read()

        Image.open(
            io.BytesIO(raw)
        ).convert('RGB')

    except UnidentifiedImageError:

        flash("Invalid image format.")

        return redirect(url_for('index'))

    except Exception as e:

        flash(f"Error: {e}")

        return redirect(url_for('index'))

    # predict

    try:

        classname, confidence = (
            predict_tflite_from_bytes(raw)
        )

    except Exception as e:

        flash(f"Inference error: {e}")

        return redirect(url_for('index'))

    # LOW CONFIDENCE FILTER

    if confidence < 0.50:

        classname = "Unknown Disease"

    # SAVE TO DATABASE

    conn = sqlite3.connect('data.db')

    c = conn.cursor()

    c.execute(
        "INSERT INTO predictions (class, confidence) VALUES (?, ?)",
        (classname, confidence)
    )

    conn.commit()
    conn.close()

    # disease info

    info = DISEASE_INFO.get(classname, {
        "treatment": "N/A",
        "prevention": "N/A",
        "fertilizer": "N/A"
    })

    # mime

    mime = 'image/jpeg'

    if raw.startswith(b'\x89PNG'):

        mime = 'image/png'

    elif raw.startswith(b'RIFF'):

        mime = 'image/webp'

    # encode image

    b64 = base64.b64encode(raw).decode('utf-8')

    data_uri = f"data:{mime};base64,{b64}"

    return render_template(
        'result.html',
        classname=classname,
        confidence=round(confidence * 100, 2),
        info=info,
        image_data=data_uri
    )

# =====================================
# DASHBOARD
# =====================================

@app.route('/dashboard')
def dashboard():

    conn = sqlite3.connect('data.db')

    c = conn.cursor()

    c.execute("""
        SELECT class, COUNT(*)
        FROM predictions
        GROUP BY class
    """)

    data = c.fetchall()

    conn.close()

    labels = [row[0] for row in data]

    values = [row[1] for row in data]

    total = sum(values)

    most_detected = "N/A"

    if values:

        max_index = values.index(max(values))

        most_detected = labels[max_index]

    return render_template(
        'dashboard.html',
        labels=labels,
        values=values,
        total=total,
        most_detected=most_detected
    )

# =====================================
# RUN
# =====================================

if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )