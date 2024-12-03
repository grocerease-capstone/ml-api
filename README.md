## ML API

### Cara Menjalankan

Untuk dapat menjalankan ML API pastikan sudah menyiapkan virtual environment. Selanjutnya install beberapa packages yang diperlukan menggunakan perintah `pip install -r requirements.txt`. ML API dijalankan menggunakan fastapi, untuk itu perlu menginstall package fastapi terlebih dahulu `pip install "fastapi[standard]"`.

Informasi lanjutan:

- Versi python: 3.10
- Command: `fastapi run main.py --host 127.0.0.1 --port 8080`

### Product Item Labels

- 0: makanan
- 1: kecantikan
- 2: home-living
- 3: minuman
- 4: product-segar
- 5: kesehatan
- 6: lainnya (jika prediksi < threshold)
