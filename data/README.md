# Folder `data/`

Dataset **tidak disertakan** di repo ini.

1. Unduh **Gender Classification Dataset** dari Kaggle:
   https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
   (cek lisensi dan ketentuan penggunaannya di halaman tersebut).
2. Ekstrak sehingga strukturnya seperti ini:

```
data/
├── Training/
│   ├── female/   (23.243 gambar)
│   └── male/     (23.766 gambar)
└── Validation/
    ├── female/   (5.841 gambar)
    └── male/     (5.808 gambar)
```

Notebook mencari folder `Training/` + `Validation/` secara otomatis. Kalau datasetmu ada di lokasi lain,
ubah `SOURCE_DIR` di cell dataset pada notebook.

Format gambar yang dibaca: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` (file `.webp` / `.jfif` diabaikan Keras).
