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
- Setelah itu, melakukan Reduksi Dimensi dengan metode PCA
- Melakukan proses Train-test-Split dengan rentang 90:10
- melakukan Standarisasi data train dan data test 

2. Setelah melakukan persiapan data, maka dilakukan dengan modelling dan evaluasi modelling dari *machine learning* dengan menggunakan 3 algoritma yang digunakan, yaitu Algoritma K-Nearest Neighbor(KNN), Algoritma Random Forest, dan Algoritma Boosting.
   - **Algoritma K-Nearest Neighbor (KNN)**
       Algoritma KNN merupakan algoritma klasifikasi yang bekerja dengan mengambil sejumlah K data terdekat (tetangganya) sebagai acuan untuk menentukan kelas dari data baru[[5]](https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi/). Algoritma ini mengklasifikasikan data berdasarkan similarity atau kemiripan atau kedekatannya terhadap data lainnya.
         ![knn](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/a0a6fe46-61a2-40d5-99b8-c76cb170ec6d)

         Gambar 3. Konsep dan Cara Kerja Algoritma KNN


Cara Kerja dari Algoritma KNN adalah:
- Tentukan jumlah tetangga (K) yang akan digunakan untuk pertimbangan penentuan kelas.
- Hitung jarak dari data baru ke masing-masing data point di dataset.
- Ambil sejumlah K data dengan jarak terdekat, kemudian tentukan kelas dari data baru tersebut.

Data teknik Algoritma KNN, untuk menentukan jarak terdekat dari data yang sudah melakukan modelling, dapat dilakukan dengan beberapa rumus matematis metriks.
1. *Euclidean Distance*

   Euclidean Distance merupakan rumus yang paling umum digunakan untuk mengukur jarak dari data train dengan data test yang terdekat. rumus ini paling sederhana sehingga library basic Python kebanyakan menggunakan rumus dari *Euclidean Distance*. jika melihat dari 1 dimension dari data test dapat dirumuskan sebagai berikut:
   $$d(x1,x2)=\sqrt{\sum_{i=1}^n (x1_i-x2_i)^2}$$
   sedangkan jika lebih dari 1 dimensi data test dapat dirumuskan sebagai berikut :
   $$dis=\sqrt{\sum_(i=1)^n (x1_i - x2_i)^2 + (y1_i - y2_i)^2 +.....}$$
   **dengan dis merupakan nilai variable lebih dari 1 dimensi.
2. Hamming Distance

   Hamming Distance jika didefinisikan sebagai rumus yang digunakan untuk mencari 2 string yang panjang yang sama dengan tujuan untuk mencari penguukuran minimum banyaknya *subtitusi / pengganti* string untuk diubah menjadi string lain[[6]](https://www.trivusi.web.id/2022/06/jenis-distance-metric.html). rumusnya dapat kita lihat dengan rumus seperti berikut : 
       $$d(x1,x2)=\frac{1}{n}\sum_{n=1}^{n=n} |x1_i-x2_i|$$
   - **Algoritma Random Forest**
      Algoritma Random Forest adalah algoritma yang menggabungkan keluaran dari beberapa *decision tree* untuk mencapai satu hasil. Algoritma ini digunakan untuk mengklasifikasi data set dalam jumlah besar.[[7]](https://algorit.ma/blog/cara-kerja-algoritma-random-forest-2022/#:~:text=Random%20Forest%20adalah%20algoritma%20dalam,skala%20dan%20performa%20yang%20tinggi.)
     Cara kerja dari Algoritma Random Fores, yaitu klasifikasi. Random Forest bekerja dengan mencari, menggabungkan, dan memprediksi dari *decision tree* untuk mendapatkan hasil yang lebih stabil dan akurat. Random Forest yang dibangun dibangun dengan metode bagging untuk meningkatkkan hasil keseluruhan dari subset fitur yang acak.
![random forest](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/eb576cac-b401-4fed-abf0-474864e7117b)

Gambar 4. Random Forest.
   - **Algoritma Boosting**
     Algoritma Boosting adalah metode algortima yang hasil analisis didapatkan dengan cara mengurangi kesalahan dalam label yang terdapat pada data[[8]]. Model Algortima seperti ini membuat kesalahan prediksi dipengaruhi oleh dari set data train yang dilatih secara berurutan untuk meningkatkan akurasi sistem sekeluruhan.
      
![xgboost_illustration](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/426be3e3-b39a-4da9-a032-674148f01750)

Gambar 5. Boosting Algorithm
      Cara kerja dari Algoritma ini sebenarnya merupaakn re-desain dari algoritma Random Forest, yaitu menggunakan pendekatan *Tree Decision*. namun, untuk menentukan dan tingkat akurasi prediksi sangat bergantung pada seberapa besar dataset yang dilatih dengan data yang dijadikan sample. dataset yang dilatih akan melakukan latihan atau train kepada dataset sample yang mana akan membentuk seperti decision tree.

## Data Understanding
   Dalaml proyek ini, dataset yang dijadikan sebagai bahan proyek diambil dari situs Kaggle, yaitu House price Prediction[[9]](https://www.kaggle.com/datasets/shree1992/housedata). berikut langkah - langkah dari hasil proyek yang saya kerjakan :
   1. **Deskripsi Variable**
      Pada bagian ini, setelah saya melakukan Download dataset yang saya lakukan, saya kemudian melakukan ekstraksi dari file House Price Prediction. bentuk file yang saya ekstrack berbentuk *csv*. setelah itu saya simpan di */content/files/* dan dilakukan pembersihan data *(Data Cleaning)*. lalu setelah dipastikan data yang sudah dibersihkan tidak bernilai null, saya melakukan pengecekan nilai :
      
 **No**|**Column**|**Non-Null Count**|**DType**
:-----:|:-----:|:-----:|:-----:|
1|date|4600 non-null|Object|
2|price|4600 non-null|float64|
3|Bedrooms |4600 non-null|float64|
4|Bathrooms|4600 non-null|float64|
5|sqft_living |4600 non-null|int64|
6|sqft_lot|4600 non-null|int64|
7|floors|4600 non-null|float64|
8|waterfront|4600 non-null|int64|
9|view|4600 non-null|int64|
10|condition|4600 non-null|int64|
11|sqft_above|4600 non-null|int64|
12|sqft_basement|4600 non-null|int64|
13|yr_built|4600 non-null|int64|
14|yr_renovated|4600 non-null|int64|
15|street| 4600 non-null | object |
16|city|4600 non-null | object |
17|statecity| 4600 non-null | object |
18|country| 4600 non-null| object |


### Variable-variable pada dataset yang ditampilkan sebagai berikut :
- date  merujuk kepada tanggal Rumah terjual
- Price merujuk kepada harga rumah yang terjual dalam kurs USD Amerika Serikat
- Bedrooms merujuk kepada jumlah kamar tidur
- Bathrooms merujuk kepada jumlah kamar mandi
- sqft_living merujuk kepada luas rumah
- sqft_lot merujuk kepada luas tanah
- waterfront merujuk kepada apakah ada air sekitar rumah
- view merujuk kepada pemandangan luar rumah
- condition merujuk kepada kondisi rumah nilaianya 1 atau 0
- sqft_above merujuk kepada luas total rumah
- sqft_basement merujuk kepada luas ruangan basement
- yr_built merujuk kepada tahun berdiri bangunan rumah
- yr_renovated merujuk kepada tahun rumah terakhir direnovasi
- street merujuk kepada jalan rumah
- city merujuk kepada kota
- statezip merujuk kepada kode pos
- country merujuk kepada negara
