# Laporan Proyek Machine Learning - Cris Yustianto Putra Tangdialla

## Daftar Isi

- [Domain Proyek](#domain-proyek)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Referensi](#referensi)

## Domain Proyek
proyek ini akan membahas mengenai prediksi harga rumah yang dataset diambil di USA yang diupdate tahun 2023 terakhir bulan maret.
![rumah](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/67ed8b28-a84f-4334-85dd-d5b7f7dcf524)
**Gambar 1. Ilustrasi Rumah pada umumnya di USA

rumah merupakan salah satu kebutuhan yang harus dimiliki orang setiap orang setelah kebutuhan sandang dan pangan sudah terpenuhi. definisi rumah menurut artikel di [[1]](https://www.rumah.com/panduan-properti/apa-itu-rumah-60592) adalah suatu bangunan tempat berpulang dari bepergian, berkegiatan, tempat untuk tidur dan beristirahat untuk memulihkan kondisi fisik dan mental yang letih dari melaksanakan tugas sehari-hari bagi penghuni. selain dari sisi bangunan, berbicara dari psikologi menurut artikel dari [[2]](https://berita.99.co/pengertian-rumah-adalah/) dapat diartikan sebagai suatu situasi lingkungan yang membuat penghuni mendapatkan kedamaian dan ketenteraman untuk memulihkan psikologi. 
dalam beberapa tahun terakhir, menurut dari artikel dari CNBC Indonesia [[3]](https://www.cnbcindonesia.com/market/20210826182211-20-271567/apa-benar-kaum-milenial-susah-punya-rumah-cek-faktanya) adalah sulit menemukan rumah idaman yang berharga terjangkau. hal ini berkaitan dengan nilai rumah yang tiap tahun yang semakin naik[[4]](https://www.fimela.com/lifestyle/read/4832551/6-hal-penting-yang-perlu-diperhatikan-saat-mencari-rumah-idaman) dan membuat calon penghuni rumah sulit untuk mengambil rumah.
![infografis-bener-ga-sih-milenial-susah-punya-rumah](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/c33bbf45-887d-4314-966d-b2f897366e81)

**Gambar 2. Penyebab Milenial Susah Punya Rumah

selain dari harga rumah yang berada di Indonesia, hal ini selaras dengan kenaikan suku bungan The Fed di USA sehingga harga rumah menjadi naik. oleh karena itu, latar belakang ini diangkat untuk membuat sebuah prediksi harga rumah yang ada di USA agar menjadi simulasi perhitungan untuk bagaimana tingkat prediksi yang bisa digunakan untuk mendapatkan prediksi harga yang cocok untuk calon penghuni rumah.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, maka diperoleh rumusan masalah pada proyek ini, yaitu:
1. Bagaimana membuat persiapan data *(Data Preparation)* Prediksi Harga Rumah di USA menggunakan *machine learning*?
2. Bagaimana cara membuat model *machine learning* untuk memprediksi harga dari Rumah di USA?

### Goals

Berdasarkan rumusan masalah di atas, maka diperoleh tujuan dari proyek ini, yaitu:
1. Untuk melakukan tahap persiapan data atau *data preparation*, agar data yang digunakan dapat dipakai untuk melatih model *machine learning*.
2. Untuk membuat model *machine learning* dalam memprediksi harga dari batu permata dengan tingkat *error* model *machine learning* yang cukup rendah.


### Solution Statements

Berdasarkan rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data atau *data preparation* dilakukan dengan menggunakan beberapa teknik persiapan data, yaitu:
- Melakukan Pembersihan data *(Data Cleaning)* dari column yang memiliki nilai korealsi yang rendah atau tidak memiliki variable parameter terkait untuk model *machine learning*.
- Melakukan Encoding Fitur Kategori pada variable kategori fitur untuk model *machine learning*
- 

2. Tahap membuat model *machine learning* untuk memprediksi harga batu permata dilakukan menggunakan model *machine learning* dengan 3 algoritma yang berbeda dan kemudian akan dilakukan evaluasi model untuk membandingkan performa model yang terbaik. Algoritma yang akan digunakan, yaitu Algoritma K-Nearest Neighbor, Algoritma Random Forest, dan Boosting Algorithm.
