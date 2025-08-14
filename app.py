from flask import Flask, render_template, request
import pickle

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Memuat model dan vectorizer yang sudah disimpan ---
# 'rb' berarti 'read binary', mode yang diperlukan untuk file pickle
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    print("Model dan Vectorizer berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file 'model.pkl' dan 'vectorizer.pkl' ada di folder yang sama.")
    model, vectorizer = None, None

# --- Mendefinisikan Rute untuk Halaman Utama ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Jika tidak ada model, tampilkan pesan error
    if not model or not vectorizer:
        return "Error: Model atau Vectorizer tidak berhasil dimuat. Periksa file .pkl Anda."

    # Inisialisasi variabel hasil
    hasil_prediksi = None
    teks_asli = None
    probabilitas_teks = None

    # Jika pengguna mengirimkan data (metode POST)
    if request.method == 'POST':
        # Mengambil teks dari form di halaman web
        teks_asli = request.form['berita']
        
        # 1. Mengubah teks input menjadi vektor TF-IDF
        vektor_teks = vectorizer.transform([teks_asli])
        
        # 2. Melakukan prediksi menggunakan model
        prediksi = model.predict(vektor_teks)[0] # [0] untuk mengambil hasil pertama
        
        # 3. Mendapatkan probabilitas prediksi
        probabilitas = model.predict_proba(vektor_teks)[0]
        
        # 4. Menyiapkan teks hasil untuk ditampilkan
        if prediksi == 1:
            hasil_prediksi = "🔴 Terdeteksi sebagai Hoaks"
            probabilitas_teks = f"Tingkat Keyakinan (Hoaks): {probabilitas[1]:.2%}"
        else:
            hasil_prediksi = "✅ Terdeteksi sebagai Faktual"
            probabilitas_teks = f"Tingkat Keyakinan (Faktual): {probabilitas[0]:.2%}"

    # Menampilkan halaman web. Jika ada hasil, tampilkan juga hasilnya.
    return render_template('index.html', hasil=hasil_prediksi, teks_asli=teks_asli, probabilitas=probabilitas_teks)

# --- Menjalankan Aplikasi Web ---
if __name__ == '__main__':
    # debug=True agar server otomatis restart jika ada perubahan kode (hanya untuk pengembangan)
    app.run(debug=True)