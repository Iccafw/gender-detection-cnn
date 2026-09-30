# Folder `models/`

Folder ini terisi otomatis saat notebook dijalankan (tidak ikut di-commit):

| File | Isi |
|---|---|
| `model1_best.h5`, `model2_best.h5`, `model3_best.h5` | bobot terbaik tiap model (berdasarkan val_accuracy) |
| `model1_log.csv`, `model2_log.csv`, `model3_log.csv` | log training per epoch |
| `gender_model_best_overall.h5` | model terbaik dari ketiganya (dipakai untuk prediksi) |

Untuk mendapatkannya, jalankan bagian training di `gender_detection_cnn.ipynb`.
