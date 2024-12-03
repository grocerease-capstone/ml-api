## ML API

### Cara Menjalankan

Untuk dapat menjalankan ML API pastikan sudah menyiapkan virtual environment. Selanjutnya install beberapa packages yang diperlukan menggunakan perintah `pip install -r requirements.txt`. ML API dijalankan menggunakan fastapi, untuk itu perlu menginstall package fastapi terlebih dahulu `pip install "fastapi[standard]"`.

Informasi lanjutan:

- Versi python: 3.10
- Command: `fastapi run main.py --host 127.0.0.1 --port 8080`

### Product Item Labels

- 0: home living
- 1: minuman
- 2: product-segar
- 3: kecantikan
- 4: kesehatan
- 5: makanan
- 6: lainnya (jika prediksi < threshold)
