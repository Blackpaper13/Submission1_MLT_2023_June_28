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
proyek ini akan membahas mengenai prediksi harga rumah yang dataset diambil di USA.
![rumah](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/67ed8b28-a84f-4334-85dd-d5b7f7dcf524)
**Gambar 1. Ilustrasi Rumah pada umumnya di USA

rumah merupakan salah satu kebutuhan yang harus dimiliki orang setiap orang setelah kebutuhan sandang dan pangan sudah terpenuhi. definisi rumah menurut artikel di [[1]](https://www.rumah.com/panduan-properti/apa-itu-rumah-60592) adalah suatu bangunan tempat berpulang dari bepergian, berkegiatan, tempat untuk tidur dan beristirahat untuk memulihkan kondisi fisik dan mental yang letih dari melaksanakan tugas sehari-hari bagi penghuni. selain dari sisi bangunan, berbicara dari psikologi menurut artikel dari [[2]](https://berita.99.co/pengertian-rumah-adalah/) dapat diartikan sebagai suatu situasi lingkungan yang membuat penghuni mendapatkan kedamaian dan ketenteraman untuk memulihkan psikologi. 
dalam beberapa tahun terakhir, menurut dari artikel dari CNBC Indonesia [[3]](https://www.cnbcindonesia.com/market/20210826182211-20-271567/apa-benar-kaum-milenial-susah-punya-rumah-cek-faktanya) adalah sulit menemukan rumah idaman yang berharga terjangkau. hal ini berkaitan dengan nilai rumah yang tiap tahun yang semakin naik[[4]](https://www.fimela.com/lifestyle/read/4832551/6-hal-penting-yang-perlu-diperhatikan-saat-mencari-rumah-idaman) dan membuat calon penghuni rumah sulit untuk mengambil rumah.
![infografis-bener-ga-sih-milenial-susah-punya-rumah](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/c33bbf45-887d-4314-966d-b2f897366e81)

**Gambar 2. Penyebab Milenial Susah Punya Rumah

selain dari harga rumah yang berada di Indonesia, hal ini selaras dengan kenaikan suku bungan The Fed di USA sehingga harga rumah menjadi naik hal ini dapat dilihat pada berita yang dikeluarkan oleh CNBC Indonesia tahun 2022 bulan Maret[[14]](https://www.cnbcindonesia.com/mymoney/20230615123235-72-446187/bunga-the-fed-bikin-warga-amerika-makin-susah), yang membuat sektor pinjaman uang menjadi tidak menarik. selain dari Pinjaman Uang, Pinjaman Rumah atau KPR rumah mengalami masalah karena banyak orang yang mengalami masalah bagi yang mengambil pinjaman rumah selama 15 tahun keatas karena bunga yang dipatok sebesar 6.7% walaupun itu sudah turun dibandingan dengan tahun sebelumnya. Oleh karena itu, latar belakang ini diangkat untuk membuat sebuah prediksi harga rumah yang ada di USA agar menjadi simulasi perhitungan untuk bagaimana tingkat prediksi yang bisa digunakan untuk mendapatkan prediksi harga yang cocok untuk calon penghuni rumah.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, maka diperoleh rumusan masalah pada proyek ini, yaitu:
1. Bagaimana membuat persiapan data *(Data Preparation)* Prediksi Harga Rumah di USA menggunakan *machine learning*?
2. Bagaimana cara membuat model *machine learning* untuk memprediksi harga dari Rumah di USA?

### Goals

Berdasarkan rumusan masalah di atas, maka diperoleh tujuan dari proyek ini, yaitu:
1. Untuk melakukan tahap persiapan data atau *data preparation*, agar data yang digunakan dapat dipakai untuk melatih model *machine learning*.
2. Untuk membuat model *machine learning* dalam memprediksi harga dari Harga Rumah di USA dengan tingkat *error* model *machine learning* yang cukup rendah dan akurasi yang paling tinggi.


### Solution Statements

Berdasarkan rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data atau *data preparation* dilakukan dengan menggunakan beberapa teknik persiapan data, yaitu:
- Melakukan Pembersihan data *(Data Cleaning)* dari column yang memiliki nilai korealsi yang rendah atau tidak memiliki variable parameter terkait untuk model *machine learning*.
- Melakukan Encoding Fitur Kategori pada variable kategori fitur untuk model *machine learning*
- Setelah itu, melakukan Reduksi Dimensi dengan metode PCA
- Melakukan proses Train-test-Split dengan rentang 80:20
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
     Algoritma Boosting adalah metode algortima yang hasil analisis didapatkan dengan cara mengurangi kesalahan dalam label yang terdapat pada data[[8]](http://etd.repository.ugm.ac.id/penelitian/detail/211569). Model Algortima seperti ini membuat kesalahan prediksi dipengaruhi oleh dari set data train yang dilatih secara berurutan untuk meningkatkan akurasi sistem sekeluruhan.
      
![xgboost_illustration](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/426be3e3-b39a-4da9-a032-674148f01750)
        Gambar 5. Boosting Algorithm
        
Cara kerja dari Algoritma ini sebenarnya merupaakn re-desain dari algoritma Random Forest, yaitu menggunakan pendekatan *Tree Decision*. namun, untuk menentukan dan tingkat akurasi prediksi sangat bergantung pada seberapa besar dataset yang dilatih dengan data yang dijadikan sample. dataset yang dilatih akan melakukan latihan atau train kepada dataset sample yang mana akan membentuk seperti decision tree. rumus untuk mencari Komputasi final dari algoritma Boosting adalah : 
![18-09-22 Ade 6](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/772999d0-ffca-4e2e-b7df-02f54beb3468)
Gambar 6. rumus dari Algortima Boosting secara general
      
## Data Understanding
***
   Dalaml proyek ini, dataset yang dijadikan sebagai bahan proyek diambil dari situs Kaggle, yaitu House price Prediction[[9]](https://www.kaggle.com/datasets/shree1992/housedata) tahun 2014. berikut langkah - langkah dari hasil proyek yang penulis kerjakan :
   1. **Cara mengambil datanya**
    Dataset yang penulis gunakan merupakan dataset yang sudah disediakan oleh kaggle yang mana data yang ada didalam dataset akan terus dilakukan update jika kita sering mendownload file bernama data.csv dari folder housedata. kemudian penulis lakukan proses import folder housedata untuk dilakukan ekstract yang mana disimpan pada folder *files*.
    Hasil dari dalam data.csv yang penulis jabarkan sebagai berikut : 
        - Dataset Characteristic: Multivariate
        - Assouciated Task : Prediction, Regression
        - Number of Instances : 4600
        - Number of Attributes : 18
        - Missing Values : N/A
        - Area : House Transaction / Property.
   2. **Deskripsi Variable**
      Pada bagian ini, setelah penulis melakukan Download dataset yang penulis lakukan, penulis kemudian melakukan ekstraksi dari file House Price Prediction. bentuk file yang penulis ekstrack berbentuk *csv*. setelah itu penulis simpan di */content/files/* dan dilakukan pembersihan data *(Data Cleaning)*. lalu setelah dipastikan data yang sudah dibersihkan tidak bernilai null, penulis melakukan pengecekan nilai :
      Tabel 1. Tabel dataset pada data.csv
         **No**|**Column**|**Non-Null Count**|**Type Data**
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
setelah dilakukan pengambilan data pada dataset data.csv, dilakukan indentifikasi variable yang dalam data.csv tersebut. hasilnya, berikut variable-variable pada dataset data.csv yang ditampilkan sebagai berikut :
        - date  dapat didefinisikan sebagai object yang merujuk kepada tanggal Rumah terjual tersebut 
        - Price dapat didefinisikan sebagai harga rumah dalam bentuk USD dollar yang tipe data adalah float64. Price ini merujuk kepada harga rumah yang terjual dalam kurs USD Amerika Serikat
        - Bedrooms dapat didefinisikan sebagai jumlah kamar pada satu rumah. tipe data adalah float64 yang mana merujuk kepada jumlah kamar tidur. keluaran data adalah 0,1,2, atau 3.
        - Bathrooms merujuk kepada jumlah kamar mandi. tipe data adalah float64 yang keluaran data berbentuk 0,1,2,atau 3. 
        - sqft_living merujuk kepada luas rumah yang keluaran data yang tipe data adalah int64. 
        - sqft_lot merujuk kepada luas tanah keluaran data yang tipe data adalah int64. 
        - waterfront merujuk kepada apakah ada air sekitar rumah keluaran data yang tipe data adalah float64. 
        - view merujuk kepada pemandangan luar rumah keluaran data yang tipe data adalah int64. 
        - condition merujuk kepada kondisi rumah yang tipe datanya adalah int64 nilaianya 1 (Ada view) atau 0 (tidak ada view)
        - sqft_above merujuk kepada luas total rumah keluaran data yang tipe data adalah int64. 
        - sqft_basement merujuk kepada luas ruangan basement keluaran data yang tipe data adalah int64. 
        - yr_built merujuk kepada tahun berdiri bangunan rumah keluaran data yang tipe data adalah int64. 
        - yr_renovated merujuk kepada tahun rumah terakhir direnovasi keluaran data yang tipe data adalah int64. 
        - street merujuk kepada jalan rumah keluaran data yang tipe data adalah object. data yang ditampilkan adalah list nama-nama jalan pada rumah yang terdaftar di data.csv
        - city merujuk kepada kota keluaran data yang tipe data adalah object. keluaran adalah nama-nama kota di USA 
        - statezip merujuk kepada kode pos keluaran data yang tipe data adalah object. keluaran adalaah kodepos USA pada rumah terdaftar.
        - country merujuk kepada negara keluaran data yang tipe data adalah object.keluaran nilai adalah USA. 

### Handle Missing Value
dengan menggunakan 3 cara, yaitu mencari variable yang mempengaruhi nilai pada harga rumah, yaitu harga, kamar tidur, kamar mandi, ruang tamu, luas tanah dan bangunan dan air. dengan menggunakan syntak (rumah.waterfront == 0).sum() pada masing-masing variable hasilnya nilainya tidak mengeluarkan null alias tidak ada nilai missing. namun untuk lebih menyakinkan penulis menggunakan syntak *isnull().sum()* dan didapatkan bahwa dataset pada data.csv tidak ada nilai null.

### EDA - Handling Outliers 
setelah melakukan data understanding, selanjutnya adalah melakukan outliers terhadap data yang memiliki tingakt reduksi yang tinggi. hasilnya outliers seperti berikut:
- Price </br>
  ![price_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/0eea8994-75a6-455d-b73d-e012bc9efeec) </br>
  Gambar 7. EDA pada Price </br>
- Bedrooms </br>
![bedrooms_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/fc9183c6-9f15-4b58-a628-557fd2b81fa2) </br>
Gambar 8.EDA pada Bedrooms </br>
- bathroom </br>
![bathrooms](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/e0f65f75-dc92-4543-a2f8-8263ceb57630) </br>
Gambar 9. EDA pada Bathrooms </br>
- sqft_living </br>
![sqft_living_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/ec4bcd82-903a-4558-8f31-21b90ea3a102) </br>
Gambar 10. EDA pada sqft_living </br>
- sqft_lot </br>
![sqft_lot](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/6c91111a-de63-484b-90a9-b2ccc7e31e24) </br>
Gambar 11. EDA pada sqft_lot </br>
- sqft_above </br>
![sqft_above_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/b6eaa045-938b-4326-bacd-00e985c29cd7) </br>
Gambar 12. EDA pada sqft_above </br>
untuk menghilangkan residu dari dataset diatas, maka menggunakan metode IQR untuk menghasilkan tingkat residu dataset yang bisa ditoleransi.
hasil dari IQR dapat dilihat sebagai berikut: </br>
![IQR hasil](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/c2037817-c858-4809-b936-154865d91df4) </br>
Gambar 13. Perhitungan IQR method. 

### EDA - Univariate Analysis
Proses univariate data analysis pada masing-masing fitur dan numerik.
   - Categorical features
     yang termasuk kedalam categorical features adalah kategori yang bernilai object. pada hasil ini, nilai dari street, date, dan statezip memiliki nilai *unique* yang terlalu banyak. namun penulis biarkan dahulu untuk tidak menghapusnya karena jumlah dataset yang masih dibawah 4000.
   - Numerical features
     yang termasuk kedalam numerical features adalah kategori yang berniali selain object, bisa berbentuk int64 atau float64. pada nilai seperti condition, view, dan waterfront perlu dihapus karena tidak mempengaruhi dan bukan key dari mencari sebuah rumah yang sesuai.
hasil dari univariate analysis dapat dilihat pada gambar berikut ini: </br>
![cat dan num](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/1702c4c5-547e-4a10-b6fb-091acd4f148b) </br>
Gambar 14. Cat dan Num Features

selanjutnya dilakukan analisis numeric features distribution dari array numeric, hasilnya bahwa nilai distribusi pada price, sqft_living, sqft_lot, dan sqft_above memiliki distribusi yang cukup baik, yang mana dapat kita lihat pada gambar berikut : 
![pairplot](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/7923fccd-3619-48f8-aaef-abb9df7ce4ba)</br>
Gambar 15. Distribusi nilai pada numeric features.

sehingga dapat kita simpulkan bahwa untuk mencari korelasi yang terbaik dapat menggunakan price dahulu untuk mencari korelasi dengan categorical features.

### Correlation Matrix: 
setelah mendapatkan nilai distribusi nilai numeric yaitu price, selanjutnya kita melihat apakah nilai price memiliki korelasi terhadap kepada object yang masuk categorical features. hasilnya dapat dilihat pada gambar berikut (penulis tampilkan pengaruh price terhadap object date, street, city, statezip, dan country)
</br>
![pengaruh price dengan date dan street](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/550f8d94-d2f8-4809-a59f-1e1b989ff703)</br>
Gambar 16. Rata-rata Price dengan date dan street </br>
![pengaruh price dengan city dan statezip](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/012b279e-7dcc-4261-8d58-bcb37320adb2)
Gambar 17. Rata-rata price dengan city dan statezip
</br>

kemudian dari hasil ini dapat dikatakan bahwa price memiliki cukup pengaruh dengan kelima objek yang pada dictonary categorical_fitur. lalu kita melihat apakah bagian dictonary numerik_fitur memiliki nilai pengaruh yang berkaitan. hasil ini dapat dijabarkan dengan memakai pairplot. namun untuk lebih sederhana dan dapat dipahami, penulis menggunaakn metode *correlation_matrix*. hasilnya dapat dilihat pada gambar :
berikut :
![matrikx](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/f628cac5-f5d2-4942-9338-04b768a33112) </br>
gambar 18. Matrix Korelasi dari numeric fitur
hasil dari *correlation matrix* dapat dijelaskan sebagai berikut :
- nilai pada yr_renovated dan sqft_basement merupakan nilai korelasi yang paling rendah, sehingga tidak terpengaruh terhadap prediksi harga rumah.
- sqft_living memiliki nilai korelasi yang tinggi dengan sqft_above. berarti untuk memprediksi harga rumah sangat berpengaruh dengan luang ruangan pada rumah dengan luas total bangunan.
- pada bagian bedrooms, bathrooms, floors, dan yr_built memiliki nilai korelasi yang cukup tinggi dan dapat dijadikan sebagai pendukung dari prediksi harga rumah.

## Data Preparation
***
Setelah mengetahui dari nilai korelasi yang sudah didapatkan, lalu kita akan menyiapkan data dari dataset cat_fitur untuk dilakukan Encoding dengan metode *One Hot Encoding*. setelah itu, dilakukan Reduksi dimensi dari dengan PCA. karena sqft_above dengan sqft_living memiliki korelasi yang tinggi, sehingga dapat dilakukan Reduksi dimensi. 
### Encoding Fitur Kategori
fungsi dari Encoding Fitur Kategori adalah untuk mendapatkan fitur baru yang sesuai dengan agar dapat terwakili dari variable kategori. karena terdapat 5 object pada categorical_fitur, oleh karena itu perlu dilakukan Encoding dengan teknik *one-hot-encoding*. hasilnya dapata dilihat pada gambar dibawah ini : </br>
![adawdadaw](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/a1d07375-484d-4f47-b1a9-f376dac1128a)
</br>
gambar 19. Hasil Encoding dengan teknik *one-hot-encoding*
### Reduksi dimensi dengan PCA *(Principal Component Analysis)*
Pada bagian ini, karena sqft_above memiliki korelasi yang tinggi dengan sqft_living sehingga perlu dilakukan pemeriksaan apakah kedua korelasi tersebut memiliki data yang redundant / data yang hasil dapat dilihat pada gambar berikut :
![Capture](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/59045bc0-7c6b-45f2-a59f-1fdb9cebdfca)
![Capture1](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/c4726fd9-c28e-483d-950c-7cee132479f2)
</br>
Gambar 20. PCA antara sqft_living dengan sqft_above dan hasil perhintungan rasio variasi dari sqft_living dengan sqft_above
### Train-Test-Split
Setelah melakukan reduksi dimensi, dilanjutkan dengan melakukan standarisasi dengan tujuan agar mempersiapkan model dengan mengurangkan mean lalu dibagi dengan standar deviasi untuk menggeser distribusi. terakhir, melakukan train, test, dan split yang mana model ini terdiri atas 80:20 (train : test) yang berasal dari hasil standarisasi yang sudah dipersiapkan sebelumnya. hasilnya adalah untuk sample total yaitu 3692 :
- dataset yang masuk kedalam whole dataset sebanyak 3692
- dataset yang masuk kedalam train sebanyak 2953
- dataset yang masuk kedalam test sebanyak 739.
### Standarisasi
Fungsi dari Standarisasi adalah untuk memastikan setelah melakukan Encoding, PCA, dan Train-Test-Split, dapat dilakukan transformasi dalam mempersiapkan pemodelan *machine learning* untuk tipe data numerik dengan mengurangkan mean lalu membaginya dengan standar deviasi untuk mencari pergeseran distribusi. Cara ini menggunakan StandardScaler. Diharapkan untuk membuat X_train yang akan digunakan untuk sebagai dataset train. hasilnya sebagai berikut : <br>
![adawd](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/cbc2ed63-062a-4b27-861a-b32ea2fbcc31)
<br>
Gambar 21. Standarisari dari numeric_features setelah dilakukan train-test-split.

## Modeling
***
Pada tahap ini, penulis menggunakan 3 metode Algoritma untuk melakukan prediksi, yaitu K-Nearest Neighbor (KNN), Random Forest, dan Algortima Boosting. berikut penjelasan dari ketiga metode algoritma ini:
- Algoritma K-Nearest Neighbor (KNN)
Algoritma K-Nearest Neigbor merupakan algoritma yang termasuk kategori algoritma *supervised learning* paling mudah dipelajari. output yang dihasilkan dari algortma ini adalah model yang diklasifikasikan berdasarkan kategori k-tetangga terdekat. tujuan utama ada algoritma ini adalah untuk mengklasifikasikan object yang di lakukan testing dari sample-sample dari training data. dalam kasus ini, parameter yang digunakan adalah parameter `n-neighbors` dengan nilai k = 20. dalam fungsinya dalam python dapat ditulis seperti berikut : 
  ```python
   model_knn = KNeighborsRegressor(n_neighbors=20)
   ```
     Kelebihan dalam menggunakan algoritma KNN[[10]](https://www.fimela.com/lifestyle/read/4832551/6-hal-penting-yang-perlu-diperhatikan-saat-mencari-rumah-idaman) adalah untuk kasus membuat prediksi harga rumah adalah:
        1. Algortima ini merupakan salah satu yang paling mudah digunakan, hanya membutuhkan k dan *distance formula*
        2. cepat melakukan dan waktu kalkulasi yang hasilnya cukup tinggi dan cukup akurat. sejak itu algoritma ini sering disebut sebagai algoritma untuk pemalas pembelajar atau hanya dibutuhkan untuk cepat selesai.
        3. Serbaguan, dapat digunakan untuk mencari regresi dan prediksi.
        namun salah satu permasalah yang sering terjadi dalam memakai algoritma KNN adalah semakin besar dataset yang akan dilakukan perhitungan dan model, maka sumber daya komputasi akan semakin besar dan waktu yang diperlukan juga akan lama.

- Algoritma Random Forest
Algorima ini merupakan sebuah inovasi pengembangan yang menjawab permasalahan dari algoritma KNN tersebut. bedanya, jika KNN menggunakan dataset yang menghasilkan model k terdekat, algoritma ini lain pendekatannya. Random Forest / hutan yang acak seperti namanya, untuk memprediksi dari sebuah dataset menggunakan *tree* atau pohon yang mengeluarkan prediksi kelas. dari prediksi ini akan mencari kandidat prediksi yang menghasilkan akurasi yang lebih tinggi sehingga tidak terjadi overfitting / prediksi yang diluar jangkauan. algoritma ini sering disebut sebagai algoritma peramal karena algoritma ini banyak digunakan untuk prediksi keinginan para pembeli walau orang tersebut tidak melihatnya lagi atau membaca keinginan pembeli[[11]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html).
Dalam Algoritma ini, parameter yang saya gunakan untuk memakai fungsi dari rumus python adalah :
    ```python
    model_RF = RandomForestRegressor(n_estimators=45, max_depth=16, random_state=55, n_jobs=-1)
    ```
    Dengan :
    - n_estimator merupakan jumlah tree sebanyak 45 trees, 
    - max-dept adalah nilai maksimal kedalama dari n_estimator sebesar 16,
    - random_state merupakan nilai acak pada dataset train sebesar 55 dan 
    - n-jobs yang bernilai -1 yang artinya adalah pekerjaan dilakukan secara paralel.

    Kelebihan dari algoritma Random Forest adalah untuk menjadi salah satu pemakaian dalam prediksi harga rumah adalah : 
        1. Karena mengambil kandidat yang terbaik, jadi untuk outlier data, ini merupakan salah satu yang terbaik
        2. overfitting risk lebih rendah
        3. lebih efisien dalam kumpulan data besar
        4. akurasi lebih tinggi dibandingkan dengan algoritma lainnya
    Namun, Kelemahan pada algoritma ini terletak pada tingkat akurasi yang terkadang terjadi *overfitting* yang beberapa kasus membuat tingkat akurasi menjadi bias. Hal ini terjadi karena pada saat klasifikasi untuk attribut dataset, sering terjadi perbedaan jumlah level yang berbeda sehingga attribut yang berperan besar pada dataset ini tidak bisa diandalkan.

- Algoritma Boosting
Pada algoritma ini terdapat 2 jenis Boosting, yaitu AdaBoost dan GradientBoost. AdaBoost *(Adaptive Boosting)* adalah algoritma yang pendekatannya memanfaatkan bagging dan boosting untuk mengembangkan peningkatan akurasi prediktor[[12]](https://dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari). algoritma ini sangat mirip seperi random forest namun bedanya, forest yang dibangun adalah stumps atau pohon yang terbuat dari 1 cabang dan 2 anakan daun serta tidak memiliki bobot kandidat yang sama pada prediksinya. semakin kecil stumps, maka nilai kemungkinan akan semakin kecil.Sedangkan GradientBoost adalah sebuah algoritma yang mengandalkan *weak learner* atau bisa dikatakan sebagai model yang lemah untuk melakukan koreksi dan perbaikan prediksi sebelumnya dengan memperhitungkan kesalahan pada prediksi tersebut[[13]](https://www.trivusi.web.id/2023/03/algoritma-gradient-boosting.html). Jadi bisa dikatakan sebagai algoritma koreksi yang memakai model - model yang dianggap lemah untuk dilakukan koreksi. Cara kerja dari Algoritma Gradient Boost adalah menggabungkan beberapa model pada tree yang lemah untuk menjadi sebuah model yang baru dan lebih akurat. bedanya dengan AdaBoost  adalah pada tree, terdapat cabang-cabang yang mengeluarkan nilai yang lemah tapi tetap digunakan untuk membuat prediksi yang baru.
Pada algoritma ini, saya menggunakan AdaBoost untuk dijadikan sebagai salah satu algoritma yang penulis gunakan. Rumus python yang digunakan adalah :
    ```python
    model_boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)
    ```
    Dengan :
    - n_estimators adalah trees yang saya gunakan adalah 50, 
    - learning_rate adalah nilai *regressor* saat boosting adalah 0.05
    - random_state adalah nilai acak pada dataset sebesar 55.

    Kelebihan pada Algoritma ini adalah : 
    - Algoritma ini merupakan salah satu algoritma yang memiliki tingkat akurasi yang tinggi.
    - data yang diperlukan tidak kompleks dan persyaratan khusus
    - kecepatan komputasi yang tinggi walau data yang diperlukan banyak.

    Namun dari kelebihan tersebut, muncul masalah yang ada pada algoritma boosting ini :
    - karena tingkat akurasi yang tinggi, maka diperlukan kalkulasi dan perhitungan yang cermat
    - mudah mengalami overfitting  / model yang prediksi di luar jangkauan
    - data yang diperlukan sangat besar

    Dari ketiga Algoritma yang sudah penulis jabarkan baik kelebihan dan kekurangan pada masing-masing algoritma, diharapkan hasil dari ketiga model tersebut dapat mencari manakah model algoritma yang terbaik untuk digunakan.
    
## Evaluation
***
pada bagian ini, penulis melakukan evaluasi dari ketiga algoritma yang penulis gunakan untuk dilakukan prediksi dari ketiga algoritma tersebut, manakah yang merupakan algoritma yang paling mendekati dari prediksi dari hasil test sebenarnya. 
pertama, penulis menggunakan rumus MSE atau *Mean Squared Error* yang berguna untuk menjumlahkan selisih kuadrat rata-rata nila sebenarnya dengan nilai prediksi. rumus MSE dapat dinyatakan sebagai berikut : 

$$MSE=\frac{1}{n}\sum_{i=1}^{n} (A_t-F_t)^2$$
dimana : 
A_t = Nilai aktual permintaan
F_t = Nilai hasil Prediksi
n = Banyaknya data.

Hasil dari perhitungan MSE dapat dilihat pada gambar berikut : 
<br>
![adawd fa](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/1ffccb9c-e4f3-4c7b-a4ad-aa795e64aeb2)<br>
gambar 22. Hasil perhitungan MSE pada train dan test pada KNN, Random Forest,dan Boosting

Dengan rumus diatas, dapat kita memprediksi bahwa semakin kecil nilai regresi dari MSE, maka semakin baik model regresi untuk mempredksi nilai target.
Hasil dair MSE dapat dilihat pada gambar yang bawah ini : <br>
![dadawdawd](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/4be57ce2-7a5c-4cc6-8c3f-f936ec7a44cc)<br>
Gambar 23. Hasil menggunakan MSE untuk 3 algoritma yang sudah dilakukan evaluasi.

dari hasil evaluasi dari 3 algoritma tersebut, Algoritma Random Forest merupakan Algoritma yang membuat model yang mendekati dengan nilai prediksi sebenarnya.
hasil dari ketiga algoritma tersebut dapat dilihat pada tabel berikut ini :<br> 

![adawdaw](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/80923a92-0372-47ad-ac0a-0d3ad5f7dc40)<br>
Gambar 24. Tabel hasil Evaluasi dari 3 algoritma 

terakhir, dilakukan pencarian prediksi dari hasil evaluasi dari ketiga Algoritma, perlu mencari manakah tingkat akurasi yang paling tinggi dari ketiga algoritma yang sudah dijalankan. hasilnya dapat dilihat pada gambar berikut : <br>
![dafgsfe](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/433a551f-7929-4fe4-ab48-0446ad3bb342)<br>
Gambar 25. Hasil Akhir prediksi dan nilai akurasi dari KNN, RandomForest, dan Boosting.

Hasil dari 10 dataset yang penulis tampilkan dan nilai akurasi yang sudah ditampilkan, Algoritma Random Forest merupakan algoritma dengan tingkat akurasi paling tinggi, sekitar 68,411% disusul dengan K-Nearest Neighbor sebesar 59,055%. secara *goals* tujuan yang diinginkan sudah tercapai. Namun, beberapa catatan terutama peningkatan akurasi pada ketiga algoritma perlu dilakukan di masa akan datang.

### Referensi
***
​​[1]	“Analisis Klasifikasi Menggunakan Metode Gradient Boosting Machine (GBM) dan Light Gradient Boosting Machine (LGBM).” http://etd.repository.ugm.ac.id/penelitian/detail/211569 (diakses 6 Juli 2023). 
​[2]	“House price prediction | Kaggle.” https://www.kaggle.com/datasets/shree1992/housedata (diakses 6 Juli 2023). 
​[3]	“What’s the KNN?. Understanding the Lazy Learner… | by Jisha Obukwelu | Nerd For Tech | Medium.” https://medium.com/nerd-for-tech/whats-the-knn-74e84458bd24 (diakses 6 Juli 2023). 
​[4]	“Algoritma Random Forest: Pengertian dan Kegunaannya - Trivusi.” https://www.trivusi.web.id/2022/08/algoritma-random-forest.html (diakses 6 Juli 2023). 
​[5]	“Algoritma Machine Learning yang Harus Kamu Pelajari di Tahun...” https://dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari (diakses 6 Juli 2023). 
​[6]	“Gradient Boosting: Pengertian, Cara Kerja, dan Kegunaannya - Trivusi.” https://www.trivusi.web.id/2023/03/algoritma-gradient-boosting.html (diakses 6 Juli 2023). 
​[7]	“Rumah Adalah Bangunan Tempat Tinggal. Lalu, Apa saja Fungsinya?” https://berita.99.co/pengertian-rumah-adalah/ (diakses 6 Juli 2023). 
​[8]	“Apa Itu Rumah? Ini Penjelasannya dari Berbagai Aspek dan 5 Fungsinya.” https://www.rumah.com/panduan-properti/apa-itu-rumah-60592 (diakses 6 Juli 2023). 
​[9]	“Apa Benar Kaum Milenial Susah Punya Rumah? Cek Faktanya.” https://www.cnbcindonesia.com/market/20210826182211-20-271567/apa-benar-kaum-milenial-susah-punya-rumah-cek-faktanya (diakses 6 Juli 2023). 
​[10]	“6 Hal Penting yang Perlu Diperhatikan Saat Mencari Rumah Idaman - Lifestyle Fimela.com.” https://www.fimela.com/lifestyle/read/4832551/6-hal-penting-yang-perlu-diperhatikan-saat-mencari-rumah-idaman (diakses 6 Juli 2023). 
​[11]	“Algoritma K-Nearest Neighbor (KNN) untuk Klasifikasi - IlmudataPy.” https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi/ (diakses 6 Juli 2023). 
​[12]	“Pengertian dan Jenis-jenis Distance Metric pada Machine Learning - Trivusi.” https://www.trivusi.web.id/2022/06/jenis-distance-metric.html (diakses 6 Juli 2023). 
​[13]	“Cara Kerja Algoritma Random Forest - Algoritma.” https://algorit.ma/blog/cara-kerja-algoritma-random-forest-2022/ (diakses 6 Juli 2023). 
​[14]	“Bunga The Fed Bikin Warga Amerika Makin Susah.” https://www.cnbcindonesia.com/mymoney/20230615123235-72-446187/bunga-the-fed-bikin-warga-amerika-makin-susah (diakses 6 Juli 2023). 
