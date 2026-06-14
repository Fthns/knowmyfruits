import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
from pillow_heif import register_heif_opener


app = Flask(__name__)
app.secret_key = "kunci_rahasia_buah_anda"

# Daftar ekstensi yang didukung
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'jfif', 'heif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_input(img_array):
    
    img_array = img_array.astype('float32') / 255.0
    
    # Nilai mean dan std di-reshape menjadi (1, 1, 1, 3) agar sesuai dengan dimensi batch [batch, height, width, channels]
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))
    
    # Rumus normalisasi: (img - mean) / std
    img_array = (img_array - mean) / std
    return img_array

# --- LOAD MODEL ---
try:
    # Membuat sub-class dari Dense untuk membuang parameter yang bikin error (Bypass quantization_config)
    from tensorflow.keras.layers import Dense
    
    class SafeDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    # Muat model dengan mendaftarkan SafeDense ke dalam custom_objects
    _model = tf.keras.models.load_model(
        'model/model_acc2.h5', 
        compile=False, 
        custom_objects={'Dense': SafeDense}
    )
    _classes = ['fresh_apple', 'fresh_banana', 'fresh_orange', 'rotten_apple', 'rotten_orange', 'rotten_banana']
    print("\n[SUKSES] Model berhasil dimuat menggunakan bypass SafeDense!\n")
except Exception as e:
    _model = None
    print(f"\n[ERROR] Gagal memuat model: {e}\n")

@app.route('/')
def index():
    return render_template('index.html', img_uploaded=None)

@app.route('/prediction', methods=['POST'])
def pred():
    img_uploaded = None
    predicted_label = ""

    if _model is None:
        flash("Sistem Error: Model AI gagal dimuat di server. Periksa log terminal.")
        return redirect(url_for('index'))

    try:
        if 'file' not in request.files:
            flash("Gagal: Tidak ada form file.")
            return redirect(url_for('index'))

        file = request.files['file']

        if file.filename == '':
            flash("Peringatan: Pilih gambar terlebih dahulu.")
            return redirect(url_for('index'))

        if file and allowed_file(file.filename):
            # Membaca byte gambar
            img_bytes = file.read()
            
            try:
                user_img = Image.open(io.BytesIO(img_bytes))
                user_img = user_img.convert('RGB')
                user_img = user_img.resize((150, 150))
                img_array = np.array(user_img)
                img_array = np.expand_dims(img_array, axis=0)
                processed_img = preprocess_input(img_array)
                raw_predictions = _model.predict(processed_img)[0]
                predicted_class_index = np.argmax(raw_predictions)
                raw_predicted_label = _classes[predicted_class_index]
                predicted_label = raw_predicted_label.replace('_', ' ').title()
                
                # Simpan gambar untuk ditampilkan kembali di web (Base64)
                img_uploaded = base64.b64encode(img_bytes).decode('utf-8')

                return render_template('index.html', img_uploaded=img_uploaded, predicted_label=predicted_label)
            
            except Exception as e:
                flash(f"Gagal memproses gambar: {str(e)}")
                return redirect(url_for('index'))
        
        else:
            flash("Format file tidak didukung. Gunakan JPG, PNG, WEBP, atau HEIC.")
            return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error Sistem: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)