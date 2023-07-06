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
      Cara kerja dari Algoritma ini sebenarnya merupaakn re-desain dari algoritma Random Forest, yaitu menggunakan pendekatan *Tree Decision*. namun, untuk menentukan dan tingkat akurasi prediksi sangat bergantung pada seberapa besar dataset yang dilatih dengan data yang dijadikan sample. dataset yang dilatih akan melakukan latihan atau train kepada dataset sample yang mana akan membentuk seperti decision tree. rumus untuk mencari Komputasi final dari algoritma Boosting adalah : 
        ![18-09-22 Ade 6](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/772999d0-ffca-4e2e-b7df-02f54beb3468)
      

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


## Variable-variable pada dataset yang ditampilkan sebagai berikut :
***
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
## EDA - Handling Outliers 
setelah melakukan data understanding, selanjutnya adalah melakukan outliers terhadap data yang memiliki tingakt reduksi yang tinggi. hasilnya outliers seperti berikut:
- Price </br>
  ![price_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/0eea8994-75a6-455d-b73d-e012bc9efeec) </br>
  Gambar 6. Price </br>
- Bedrooms </br>
![bedrooms_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/fc9183c6-9f15-4b58-a628-557fd2b81fa2) </br>
Gambar 7. Bedrooms </br>
- bathroom </br>
![bathrooms](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/e0f65f75-dc92-4543-a2f8-8263ceb57630) </br>
Gambar 8. Bathrooms </br>
- sqft_living </br>
![sqft_living_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/ec4bcd82-903a-4558-8f31-21b90ea3a102) </br>
Gambar 9. sqft_living </br>
- sqft_lot </br>
![sqft_lot](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/6c91111a-de63-484b-90a9-b2ccc7e31e24) </br>
Gambar 10. sqft_lot </br>
- sqft_above </br>
![sqft_above_sns](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/b6eaa045-938b-4326-bacd-00e985c29cd7) </br>
Gambar 11. sqft_above </br>
untuk menghilangkan residu dari dataset diatas, maka menggunakan metode IQR untuk menghasilkan tingkat residu dataset yang bisa ditoleransi.
hasil dari IQR dapat dilihat sebagai berikut: </br>
![IQR hasil](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/c2037817-c858-4809-b936-154865d91df4) </br>
Gambar 12. IQR metode 

## EDA - Univariate Analysis
***
Proses univariate data analysis pada masing-masing fitur dan numerik.
   - Categorical features
     yang termasuk kedalam categorical features adalah kategori yang bernilai object. pada hasil ini, nilai dari street, date, dan statezip memiliki nilai *unique* yang terlalu banyak. namun saya biarkan dahulu untuk tidak menghapusnya karena jumlah dataset yang masih dibawah 4000.
   - Numerical features
     yang termasuk kedalam numerical features adalah kategori yang berniali selain object, bisa berbentuk int64 atau float64. pada nilai seperti condition, view, dan waterfront perlu dihapus karena tidak mempengaruhi dan bukan key dari mencari sebuah rumah yang sesuai.
hasil dari univariate analysis dapat dilihat pada gambar berikut ini: </br>
![cat dan num](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/1702c4c5-547e-4a10-b6fb-091acd4f148b) </br>
Gambar 13. Cat dan Num Features

selanjutnya dilakukan analisis numeric features distribution dari array numeric, hasilnya bahwa nilai distribusi pada price, sqft_living, sqft_lot, dan sqft_above memiliki distribusi yang cukup baik, yang mana dapat kita lihat pada gambar berikut : 
![pairplot](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/7923fccd-3619-48f8-aaef-abb9df7ce4ba)</br>
Gambar 14. Distribusi nilai pada numeric features.

sehingga dapat kita simpulkan bahwa untuk mencari korelasi yang terbaik dapat menggunakan price dahulu untuk mencari korelasi dengan categorical features.

### Correlation Matrix: 
setelah mendapatkan nilai distribusi nilai numeric yaitu price, selanjutnya kita melihat apakah nilai price memiliki korelasi terhadap kepada object yang masuk categorical features. hasilnya dapat dilihat pada gambar berikut (saya tampilkan pengaruh price terhadap object date, street, city, statezip, dan country)
</br>
![pengaruh price dengan date dan street](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/550f8d94-d2f8-4809-a59f-1e1b989ff703)</br>
Gambar 15. Rata-rata Price dengan date dan street </br>
![pengaruh price dengan city dan statezip](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/012b279e-7dcc-4261-8d58-bcb37320adb2)
Gambar 16. Rata-rata price dengan city dan statezip
</br>

kemudian dari hasil ini dapat dikatakan bahwa price memiliki cukup pengaruh dengan kelima objek yang pada dictonary categorical_fitur. lalu kita melihat apakah bagian dictonary numerik_fitur memiliki nilai pengaruh yang berkaitan. hasil ini dapat dijabarkan dengan memakai pairplot. namun untuk lebih sederhana dan dapat dipahami, saya menggunaakn metode *correlation_matrix*. hasilnya dapat dilihat pada gambar :
berikut :
![matrikx](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/f628cac5-f5d2-4942-9338-04b768a33112) </br>
gambar 17. Matrix Korelasi dari numeric fitur
hasil dari *correlation matrix* dapat dijelaskan sebagai berikut :
- nilai pada yr_renovated dan sqft_basement merupakan nilai korelasi yang paling rendah, sehingga tidak terpengaruh terhadap prediksi harga rumah.
- sqft_living memiliki nilai korelasi yang tinggi dengan sqft_above. berarti untuk memprediksi harga rumah sangat berpengaruh dengan luang ruangan pada rumah dengan luas total bangunan.
- pada bagian bedrooms, bathrooms, floors, dan yr_built memiliki nilai korelasi yang cukup tinggi dan dapat dijadikan sebagai pendukung dari prediksi harga rumah.

## Data Preparation
***
setelah mengetahui dari nilai korelasi yang sudah didapatkan, lalu kita akan menyiapkan data dari dataset cat_fitur untuk dilakukan Encoding dengan metode *One Hot Encoding*. setelah itu, dilakukan Reduksi dimensi dari dengan PCA. karena sqft_above dengan sqft_living memiliki korelasi yang tinggi, sehingga dapat dilakukan Reduksi dimensi. hasil dapat dilihat pada gambar berikut :
![Capture](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/59045bc0-7c6b-45f2-a59f-1fdb9cebdfca)
![Capture1](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/c4726fd9-c28e-483d-950c-7cee132479f2)
</br>
Gambar 18. PCA antara sqft_living dengan sqft_above dan hasil perhintungan rasio variasi dari sqft_living dengan sqft_above

setelah melakukan reduksi dimensi, dilanjutkan dengan melakukan standarisasi dengan tujuan agar mempersiapkan model dengan mengurangkan mean lalu dibagi dengan standar deviasi untuk menggeser distribusi. terakhir, melakukan train, test, dan split yang mana model ini terdiri atas 80:20 (train : test) yang berasal dari hasil standarisasi yang sudah dipersiapkan sebelumnya. hasilnya adalah untuk sample total yaitu 3692 :
- dataset yang masuk kedalam train sebanyak 2953
- dataset yang masuk kedalam test sebanyak 739.
## Modeling
***
Pada tahap ini, saya menggunakan 3 metode Algoritma untuk melakukan prediksi, yaitu K-Nearest Neighbor (KNN), Random Forest, dan Algortima Boosting. berikut penjelasan dari ketiga metode algoritma ini:
- Algoritma K-Nearest Neighbor (KNN)
Algoritma K-Nearest Neigbor merupakan algoritma yang termasuk kategori algoritma *supervised learning* paling mudah dipelajari. output yang dihasilkan dari algortma ini adalah model yang diklasifikasikan berdasarkan kategori k-tetangga terdekat. tujuan utama ada algoritma ini adalah untuk mengklasifikasikan object yang di lakukan testing dari sample-sample dari training data. 
Kelebihan dalam menggunakan algoritma KNN adalah [[10]](https://medium.com/nerd-for-tech/whats-the-knn-74e84458bd24):
    1. Algortima ini merupakan salah satu yang paling mudah digunakan, hanya membutuhkan k dan *distance formula*
    2. cepat melakukan dan waktu kalkulasi yang relatif hasilnya tinggi dan akurat. sejak itu algoritma ini sering disebut sebagai algoritma untuk pemalas pembelajar atau hanya dibutuhkan untuk cepat selesai.
    3. Serbaguan, dapat digunakan untuk mencari regresi atau klasifikasi.

    Namun dari 3 kelebihan diatas, terdapat masalah yang sering menjadi kekurangan jika menggunakan KNN ini:
    1. Karena menggunakan sumber daya komputasi, jadi memerlukan komputasi yang tinggi, pelatihan yang data yang banyak dan memory yang besar untuk hanya membuat sebuah model *machine learning*
    2. karena data yang besar dan random jenis datanya, sehingga sangat sensitif terhadap fitur / sample data yang tidak relevan dan skala data.
    3. karena semakin banyak datanya, maka semakin lambat kinerja model *machine learning* tersebut.
    
- Algoritma Random Forest
Algorima ini merupakan sebuah inovasi pengembangan yang menjawab permasalahan dari algoritma KNN tersebut. bedanya, jika KNN menggunakan dataset yang menghasilkan model k terdekat, algoritma ini lain pendekatannya. Random Forest / hutan yang acak seperti namanya, untuk memprediksi dari sebuah dataset menggunakan *tree* atau pohon yang mengeluarkan prediksi kelas. dari prediksi ini akan mencari kandidat prediksi yang menghasilkan akurasi yang lebih tinggi sehingga tidak terjadi overfitting / prediksi yang diluar jangkauan. algoritma ini sering disebut sebagai algoritma peramal karena algoritma ini banyak digunakan untuk prediksi keinginan para pembeli walau orang tersebut tidak melihatnya lagi atau membaca keinginan pembeli[[11]](https://www.trivusi.web.id/2022/08/algoritma-random-forest.html).
    Kelebihan dari algoritma Random Forest adalah : 
    1. Karena mengambil kandidat yang terbaik, jadi untuk outlier data, ini merupakan salah satu yang terbaik
    2. algoritma ini bekerja dengan baik dengan data non-linear
    3. overfitting risk lebih rendah
    4. lebih efisien dalam kumpulan data besar
    5. akurasi lebih tinggi dibandingkan dengan algoritma lainnya

    Namun terdapat kekurangan jika menggunakan algoritma random Forest :
    1. sering terjadi Bias prediksi jika berhadapan dengan variable kategori
    2. Algoritma ini merupakan salah satu yang sering menghabiskan komputasi yang sangat besar
    3. Tidak cocok untuk mengggunakan dalam situasi dataset yang linear

- Algoritma Boosting
Pada algoritma ini terdapat 2 jenis Boosting, yaitu AdaBoost dan GradientBoost. AdaBoost *(Adaptive Boosting)* adalah algoritma yang pendekatannya memanfaatkan bagging dan boosting untuk mengembangkan peningkatan akurasi prediktor[[12]](https://dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari). algoritma ini sangat mirip seperi random forest namun bedanya, forest yang dibangun adalah stumps atau pohon yang terbuat dari 1 cabang dan 2 anakan daun serta tidak memiliki bobot kandidat yang sama pada prediksinya. semakin kecil stumps, maka nilai kemungkinan akan semakin kecil. 
sedangkan GradientBoost adalah sebuah algoritma yang mengandalkan *weak learner* atau bisa dikatakan sebagai model yang lemah untuk melakukan koreksi dan perbaikan prediksi sebelumnya dengan memperhitungkan kesalahan pada prediksi tersebut[[13]](https://www.trivusi.web.id/2023/03/algoritma-gradient-boosting.html). Jadi bisa dikatakan sebagai algoritma koreksi yang memakai model - model yang dianggap lemah untuk dilakukan koreksi. Cara kerja dari Algoritma Gradient Boost adalah menggabungkan beberapa model pada tree yang lemah untuk menjadi sebuah model yang baru dan lebih akurat. bedanya dengan AdaBoost  adalah pada tree, terdapat cabang-cabang yang mengeluarkan nilai yang lemah tapi tetap digunakan untuk membuat prediksi yang baru.

Kelebihan pada Algoritma ini adalah : 
- Algoritma ini merupakan salah satu algoritma yang memiliki tingkat akurasi yang tinggi.
- data yang diperlukan tidak kompleks dan persyaratan khusus
- kecepatan komputasi yang tinggi walau data yang diperlukan banyak.

Namun dari kelebihan tersebut, muncul masalah yang ada pada algoritma boosting ini :
- karena tingkat akurasi yang tinggi, maka diperlukan kalkulasi dan perhitungan yang cermat
- mudah mengalami overfitting  / model yang prediksi di luar jangkauan
- data yang diperlukan sangat besar

## Evaluation
***
pada bagian ini, saya melakukan evaluasi dari ketiga algoritma yang saya gunakan untuk dilakukan prediksi dari ketiga algoritma tersebut, manakah yang merupakan algoritma yang paling mendekati dari prediksi dari hasil test sebenarnya. 
pertama, saya menggunakan rumus MSE atau *Mean Squared Error* yang berguna untuk menjumlahkan selisih kuadrat rata-rata nila sebenarnya dengan nilai prediksi. rumus MSE dapat dinyatakan sebagai berikut : 

$$MSE=\frac{1}{n}\sum_{i=1}^{n} (A_t-F_t)^2$$
dimana : 
A_t = Nilai aktual permintaan
F_t = Nilai hasil Prediksi
n = Banyaknya data.

dengan rumus diatas, dapat kita memprediksi bahwa semakin kecil nilai regresi dari MSE, maka semakin baik model regresi untuk mempredksi nilai target.
Hasil dair MSE dapat dilihat pada gambar yang bawah ini : <br>
![adlawnknd](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/9b8f6dc8-6a5b-4345-ad80-48923b681dd5)<br>
Gambar 19. Hasil menggunakan MSE untuk 3 algoritma yang sudah dilakukan evaluasi.

dari hasil evaluasi dari 3 algoritma tersebut, Algoritma Random Forest merupakan Algoritma yang membuat model yang mendekati dengan nilai prediksi sebenarnya.
hasil dari ketiga algoritma tersebut dapat dilihat pada tabel berikut ini :<br> 

![adawdaw](https://github.com/Blackpaper13/Submission1_MLT_2023_June_28/assets/63518506/80923a92-0372-47ad-ac0a-0d3ad5f7dc40)<br>
Gambar 20. Tabel hasil Evaluasi dari 3 algoritma 

Namun dari 10 dataset yang saya tampilkan beberapa nilai prediksi Algoritma Random Forest tidak sesuai karena bias akan prediksi. hal tesebut wajar karena itu merupakan salah satu kelemahan dari Algoritma Random Forest. namun, dari hasil diatas masih bisa digambarkan bahwa Algoritma selain Random Forest seperti KNN dan Boosting masih dapat digunakan untuk memprediksi nilai yang mendekati dengan nilai sebenarnya aktual. 

### Referensi
***
​​[1]	“Gradient Boosting: Pengertian, Cara Kerja, dan Kegunaannya - Trivusi.” https://www.trivusi.web.id/2023/03/algoritma-gradient-boosting.html (diakses 6 Juli 2023). 

​[2]	“Algoritma Machine Learning yang Harus Kamu Pelajari di Tahun...” https://dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari (diakses 6 Juli 2023). 

​[3]	“Algoritma Random Forest: Pengertian dan Kegunaannya - Trivusi.” https://www.trivusi.web.id/2022/08/algoritma-random-forest.html (diakses 6 Juli 2023). 

​[4]	“What’s the KNN?. Understanding the Lazy Learner… | by Jisha Obukwelu | Nerd For Tech | Medium.” https://medium.com/nerd-for-tech/whats-the-knn-74e84458bd24 (diakses 6 Juli 2023). 

​[5]	“House price prediction | Kaggle.” https://www.kaggle.com/datasets/shree1992/housedata (diakses 6 Juli 2023). 

​[6]	“Cara Kerja Algoritma Random Forest - Algoritma.” https://algorit.ma/blog/cara-kerja-algoritma-random-forest-2022/ (diakses 6 Juli 2023). 

​[7]	“Pengertian dan Jenis-jenis Distance Metric pada Machine Learning - Trivusi.” https://www.trivusi.web.id/2022/06/jenis-distance-metric.html (diakses 6 Juli 2023). 

​[8]	“6 Hal Penting yang Perlu Diperhatikan Saat Mencari Rumah Idaman - Lifestyle Fimela.com.” https://www.fimela.com/lifestyle/read/4832551/6-hal-penting-yang-perlu-diperhatikan-saat-mencari-rumah-idaman (diakses 6 Juli 2023). 

​[9]	“Apa Benar Kaum Milenial Susah Punya Rumah? Cek Faktanya.” https://www.cnbcindonesia.com/market/20210826182211-20-271567/apa-benar-kaum-milenial-susah-punya-rumah-cek-faktanya (diakses 6 Juli 2023). 

​[10]	“Rumah Adalah Bangunan Tempat Tinggal. Lalu, Apa saja Fungsinya?” https://berita.99.co/pengertian-rumah-adalah/ (diakses 6 Juli 2023). 

​[11]	“Apa Itu Rumah? Ini Penjelasannya dari Berbagai Aspek dan 5 Fungsinya.” https://www.rumah.com/panduan-properti/apa-itu-rumah-60592 (diakses 6 Juli 2023). 

