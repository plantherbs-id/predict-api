import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'plantherbs-credentials.json'
storage_client = storage.Client()


def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req


# model_filename = 'my_model_fix.h5'
# model_bucket = storage_client.get_bucket('sa-lindungi-model-bucket')
# model_blob = model_bucket.blob(model_filename)
# model_blob.download_to_filename(model_filename)
# model = load_model(model_filename, custom_objects={'req': req})
model = load_model('my_model_fix.h5', custom_objects={'req': req})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            image_bucket = storage_client.get_bucket(
                'plantherbs-bucket')
            filename = request.json['filename']
            img_blob = image_bucket.blob('input/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
            img_blob1 = image_bucket.blob1('https://storage.googleapis.com/plantherbs-bucket/output/Asem%20Jawa.jpg' + 'Asam Jawa')
            img_path1 = BytesIO(img_blob1.download_as_bytes())
            img_blob2 = image_bucket.blob2('https://storage.googleapis.com/plantherbs-bucket/output/Belimbing%20Wuluh.jpeg' + 'Belimbing Wuluh')
            img_path2 = BytesIO(img_blob2.download_as_bytes())
            img_blob3 = image_bucket.blob3('https://storage.googleapis.com/plantherbs-bucket/output/Biduri.jpg' + 'Biduri')
            img_path3 = BytesIO(img_blob3.download_as_bytes())
            img_blob4 = image_bucket.blob4('https://storage.googleapis.com/plantherbs-bucket/output/Cocor%20Bebek.jpg' + 'Cocor Bebek')
            img_path4 = BytesIO(img_blob4.download_as_bytes())
            img_blob5 = image_bucket.blob5('https://storage.googleapis.com/plantherbs-bucket/output/Jambu%20Air.jpg' + 'Jambu Air')
            img_path5 = BytesIO(img_blob5.download_as_bytes())
            img_blob6 = image_bucket.blob6('https://storage.googleapis.com/plantherbs-bucket/output/Katuk.jpg' + 'Katuk')
            img_path6 = BytesIO(img_blob6.download_as_bytes())
            img_blob7 = image_bucket.blob7('https://storage.googleapis.com/plantherbs-bucket/output/Kelor.jpeg' + 'Kelor')
            img_path7 = BytesIO(img_blob7.download_as_bytes())
            img_blob8 = image_bucket.blob8('https://storage.googleapis.com/plantherbs-bucket/output/Kemangi.jpeg' + 'Kemangi')
            img_path8 = BytesIO(img_blob8.download_as_bytes())
            img_blob9 = image_bucket.blob9('https://storage.googleapis.com/plantherbs-bucket/output/Kersen.jpg' + 'Kersen')
            img_path9 = BytesIO(img_blob9.download_as_bytes())
            img_blob10 = image_bucket.blob10('https://storage.googleapis.com/plantherbs-bucket/output/Lengkuas.jpg' + 'Lengkuas')
            img_path10 = BytesIO(img_blob10.download_as_bytes())
            img_blob11 = image_bucket.blob11('https://storage.googleapis.com/plantherbs-bucket/output/Lidah%20Buaya.jpg' + 'Lidah Buaya')
            img_path11 = BytesIO(img_blob11.download_as_bytes())
            img_blob12 = image_bucket.blob12('https://storage.googleapis.com/plantherbs-bucket/output/Mimba.jpg' + 'Mimba')
            img_path12 = BytesIO(img_blob12.download_as_bytes())
            img_blob13 = image_bucket.blob13('https://storage.googleapis.com/plantherbs-bucket/output/Mint.jpeg' + 'Mint')
            img_path13 = BytesIO(img_blob13.download_as_bytes())
            img_blob14 = image_bucket.blob14('https://storage.googleapis.com/plantherbs-bucket/output/Nangka.jpg' + 'Nangka')
            img_path14 = BytesIO(img_blob14.download_as_bytes())
            img_blob15 = image_bucket.blob15('https://storage.googleapis.com/plantherbs-bucket/output/Pandan.jpg' + 'Pandan')
            img_path15 = BytesIO(img_blob15.download_as_bytes())
            img_blob16 = image_bucket.blob16('https://storage.googleapis.com/plantherbs-bucket/output/Pepaya.jpg' + 'Pepaya')
            img_path16 = BytesIO(img_blob16.download_as_bytes())
            img_blob17 = image_bucket.blob17('https://storage.googleapis.com/plantherbs-bucket/output/Saga.jpg' + 'Saga')
            img_path17 = BytesIO(img_blob17.download_as_bytes())
            img_blob18 = image_bucket.blob18('https://storage.googleapis.com/plantherbs-bucket/output/Seledri.jpg' + 'Seledri')
            img_path18 = BytesIO(img_blob18.download_as_bytes())
            img_blob19 = image_bucket.blob19('https://storage.googleapis.com/plantherbs-bucket/output/Sirih.jpg' + 'Sirih')
            img_path19 = BytesIO(img_blob19.download_as_bytes())
            img_blob20 = image_bucket.blob20('https://storage.googleapis.com/plantherbs-bucket/output/Sirsak.jpg' + 'Sirsak')
            img_path20 = BytesIO(img_blob20.download_as_bytes())
            
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond

        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images = np.vstack([x])

        # model predict
        pred_plant = model.predict(images)
        # find the max prediction of the image
        maxx = pred_plant.max()
        
        Gambar = ['img_path1', 'img_path2', 'img_path3', 'img_path4', 'img_path5', 'img_path6', 
                  'img_path7', 'img_path8', 'img_path9', 'img_path10', 'img_path11', 'img_path12', 'img_path13', 
                  'img_path14', 'img_path15', 'img_path16', 'img_path17', 'img_path18', 'img_path19', 'img_path20']

        Nama = ['Asam Jawa', 'Belimbing Wuluh', 'Biduri', 'Cocor Bebek', 'Jambu Air', 
                'Katuk', 'Kelor', 'Kemangi', 'Kersen', 'Lengkuas', 'Lidah Buaya', 'Mimba', 
                'Mint', 'Nangka', 'Pandan', 'Pepaya', 'Saga', 'Seledri', 'Sirih', 'Sirsak']
        
        Deskripsi = ['Asam jawa (Tamarindus indica) adalah termasuk tumbuhan tropis. Pohon asam tingginya dapat mencapai 30 m, dengan diameter batang sekitar 2 m. Kulit batang kasar, beralur berwarna cokelat keabu-abuan. Daunnya menyirip dengan 8-16 pasang anak daun. Di samping daging buah, banyak bagian pohon asam yang dapat dijadikan bahan obat tradisional.', 
                     'Belimbing wuluh (Averrhoa bilimbi L) merupakan salah satu spesies dalam genus Averrhoa yang tumbuh di daerah ketinggian hingga 500 m di atas permukaan laut dan dapat ditemui di tempat yang banyak terkena sinar matahari langsung tetapi cukup lembab. Ekstrak daun Belimbing wuluh mengandung kadar antioksidan yang tinggi dapat digunakan sebagai pengobatan herbal dalam mengatasi diabetes mellitus.',
                     'Biduri (Calotropis Gigantea L) adalah tanaman gulma gurun yang mampu tumbuh liar di daerah pesisir pantai sehingga untuk pembudidayaan nya sangat mudah. Tanaman ini dikenal sebagai tanaman dengan kulit akar, bunga, getah dan daun yang memiliki khasiat berbeda-beda, serta memiliki buah berisi berkas-berkas serat halus seperti sutera yang melekat pada setiap bijinya.',
                     'Cocor bebek (Kalanchoe pinnata) adalah tanaman hias yang berasal dari Genus Kalanchoe. mudah tumbuh di parit, kebun, bahkan tumbuh subur di tanah berbatu. Daunnya tebal berdaging dan mengandung banyak air, sedangkan batangnya lunak dan memiliki ruas. Warna daun hijau muda (kadang kadang abu-abu). Bunga majemuk, buah kotak. Bila dimakan cocor bebek rasanya agak asam dan dingin.', 
                     'Tanaman jambu air (Syzygium aqueumL) adalah tumbuhan dalam suku jambu-jambuan atau Myrtaceae yang berasal dari Asia Tenggara. Umumnya bagian-bagian tumbuhan jambu air berukuran lebih kecil dan kurang berbau aromatis apabila dibandingkan dengan jambu semarang. Perhatikan uraian bagian-bagian yang ditulis miring, terutama bunga dan buahnya. Daun tunggal, bertangkai pendek, letak berhadapan, daun muda berambut halus, permukaan atas daun tua licin. Helaian daun berbentuk bulat telur agak jorong, ujung tumpul, pangkal membulat, tepi rata agak melekuk ke atas, pertulangan menyirip, panjang 6-14 cm, lebar 3-6 cm, berwarna hijau.',
                     'Katuk (Sauropus androgynus) adalah spesies tumbuhan yang banyak terdapat di Asia Tenggara termasuk ke dalam genus sauropus dalam suku phyllanthaceae.[1] Tumbuhan ini dalam beberapa bahasa dikenali sebagai mani cai (马尼菜; bahasa Tionghoa), cekur manis (bahasa Melayu);[2] dan rau ngót (bahasa Vietnam).[3] Daun katuk merupakan sayuran minor yang dikenal memiliki khasiat memperlancar aliran air susu ibu (ASI). Tumbuhan ini termasuk dalam suku menir-meniran (Phyllanthaceae), dan berkerabat dengan menteng, buni, dan ceremai.', 
                     'Kelor (Moringa oleifera) ini adalah jenis tanaman dari suku Moringaceae. Tanaman kelor ini dapat tumbuh setinggi 7-11 meter. elor adalah tanaman yang bisa tumbuh dengan cepat,[3] berumur panjang,[3] berbunga sepanjang tahun,[4] dan tahan kondisi panas ekstrim. Tanaman ini berasal dari daerah tropis dan subtropis di Asia Selatan.[3] Kelor umum digunakan sebagai bahan makanan dan obat di Indonesia.[5] Biji kelor juga digunakan sebagai penjernih air skala kecil.', 
                     'Kemangi (Ocimum sanctum) merupakan tumbuhan semak dengan beberapa karakteristik (Dewi, 2007) : Tinggi antara 30-150 cm. Batang dikotil yang berkayu dengan bentuk segi empat, beralur, bercabang, berbulu, dan berwarna hijau. Bunga terdapat pada penghujung batang. Aroma daunnya khas, kuat namun lembut dengan sentuhan aroma limau. Daun kemangi merupakan salah satu bumbu bagi pepes. Sebagai lalapan, daun kemangi biasanya dimakan bersama-sama daun kubis, irisan ketimun, dan sambal untuk menemani ayam atau ikan goreng. Di Thailand ia dikenal sebagai manglak dan juga sering dijumpai dalam menu masakan setempat.', 
                     'Tanaman kersen (Muntingia calabura) adalah sejenis pohon sekaligus buahnya yang kecil dan manis berwarna merah cerah. Pohon kersen khususnya berguna sebagai pohon peneduh di pinggir jalan. Daun terletak mendatar, berseling; helaian daun tidak simetris, bundar telur lanset, tepinya bergerigi dan berujung runcing, sisi bawah berambut kelabu rapat; bertangkai pendek.', 
                     'Lengkuas (Alpinia galanga L.) merupakan jenis tumbuhan umbi-umbian yang bisa hidup di daerah dataran tinggi maupun dataran rendah. Lengkuas adalah salah satu jenis rempah-rempah yang banyak ditanam di Asia. Lengkuas dapat tumbuh di tempat yang terbuka; di bawah sinar matahari penuh atau yang sedikit terlindung. Lengkuas dapat tumbuh dengan baik di tanah yang lembab dan gembur dan akan kesulitan tumbuh di tanah yang becek. Lengkuas tumbuh subur di daerah dataran rendah sampai ketinggian 1200 meter di atas permukaan laut. Di Indonesia, lengkuas banyak ditemukan tumbuh liar di hutan jati atau di semak belukar.', \
                     'Lidah buaya (Aloe vera) adalah spesies tumbuhan dengan daun berdaging tebal dari genus Aloe. Tumbuhan ini bersifat menahun, berasal dari Jazirah Arab, dan tanaman liarnya telah menyebar ke kawasan beriklim tropis, semi-tropis, dan kering di berbagai belahan dunia. Tanaman lidah buaya banyak dibudidayakan untuk pertanian, pengobatan, dan tanaman hias, dan dapat juga ditanam di dalam pot. sejenis tanaman yang sudah dikenal sejak ribuan tahun silam dan digunakan sebagai penyubur rambut, penyembuh luka, dan untuk perawatan kulit. Aloe vera adalah tumbuhan tanpa batang atau berbatang pendek, dengan tinggi hingga 60-100 cm dan dapat berkembang biak dengan tunas.[3] Dedaunannya berdaging tebal, berwarna hijau atau hijau keabuan, dan sebagian varietas memiliki bintik putih pada permukaan batangnya.',
                     'Mimba (Azadirachta indica Juss.) adalah Tumbuh di daerah tropis, pada dataran rendah. Tanaman ini tumbuh di daerah Jawa Barat, Jawa Timur, dan Madura pada ketinggian sampai dengan 300 mdpl, tumbuh di tempat kering berkala, sering ditemukan di tepi jalan atau di hutan terang. tanaman tradisional yang bersifat antibakteri, antivirus dan antikanker. Daun mimba tersusun spiralis, mengumpul di ujung rantai, merupakan daun majemuk menyirip genap. Anak daun berjumlah genap diujung tangkai, dengan jumlah helaian 8-16. tepi daun bergerigi, bergigi, beringgit, helaian daun tipis seperti kulit dan mudah laya.', 
                     'Mint (Mentha arvensis)  adalah genus tumbuhan dalam famili Lamiaceae. Tumbuhan min tersebar luasKebanyakan tumbuh dengan baik di lingkungan yang basah dengan tanah yang lembab.', 
                     'Nangka (Artocarpus heterophyllus L) merupakan Pohon nangka umumnya berukuran sedang, sampai sekitar 20 m tingginya, walaupun ada yang mencapai 30 meter. Daun tunggal, tersebar, bertangkai 1-4 cm, helai daun agak tebal seperti kulit, kaku, bertepi rata, bulat telur terbalik sampai jorong (memanjang), 3,5-12 x 5-25 cm, dengan pangkal menyempit sedikit demi sedikit, dan ujung pendek runcing atau agak runcing.', 
                     'Pandan wangi (Pandanus ammaryllifolius Roxb.) adalah jenis tumbuhan monokotil dari famili Pandanaceae yang memiliki daun beraroma wangi yang khas. Daunnya merupakan komponen penting dalam tradisi masakan Indonesia dan negara-negara Asia Tenggara lainnya. Tumbuhan ini mudah dijumpai di pekarangan atau tumbuh liar di tepi-tepi selokan yang teduh. Akarnya besar dan memiliki akar tunggang yang menopang tumbuhan ini bila telah cukup besar. Daunnya memanjang seperti daun palem dan tersusun secara roset yang rapat, panjangnya dapat mencapai 60 cm. Beberapa varietas memiliki tepi daun yang bergerigi.', 
                     'Tanaman Pepaya yang memiliki nama latin (Carica Papaya L) merupakan tanaman yang berasal dari Meksico bagian selatan dan Nikaragua. isamping dapat diolah menjadi makanan, daun pepaya dapat pula dijadikan obat untuk beberapa jenis penyakit. Helaian daun pepaya berbentuk menyerupai tangan manusia. Apabila daun pepaya dilipat tepat di tengah, maka akan tampak bahwa daun pepaya berbentuk simetris. Rasa pahit pada daun pepaya dapat dihilangkan dengan cara merebus daun pepaya.', 
                     'Saga (Abrus precatorius) merupakan tumbuhan obat anti seriawan yang populer. Tumbuhan merambat ini, yang berbiji jingga kemerahan, juga biasa disebut sebagai saga sehingga kadang-kadang rancu dengan saga pohon. Daun tumbuhan ini jika dikombinasikan dengan daun sirih dapat digunakan menjadi obat tradisional yang ampuh mengatasi Sariawan.  Daunnya majemuk, berbentuk bulat telur serta berukuran kecil-kecil. Daun Saga bersirip ganjil dan memiliki rasa agak manis.',
                     'Seledri (Apium graveolens) memberikan rasa segar dan aroma dalam masakan. Biasanya digunakan sebagai hiasan atau bumbu dalam hidangan. Di Indonesia tumbuhan ini diperkenalkan oleh penjajah Belanda dan digunakan daunnya untuk menyedapkan sup atau sebagai lalap. Penggunaan seledri paling lengkap adalah di Eropa: daun, tangkai daun, buah, dan umbinya semua dimanfaatkan.', 
                     'Sirih (Piper betle L.) merupakan tumbuhan merambat dengan bentuk daun menyerupai jantung dan berwarna hijau. Sirih sebagai obat berbagai jenis penyakit. Daunnya yang tunggal berbentuk jantung, berujung runcing, tumbuh berselang-seling, bertangkai, dan mengeluarkan bau yang sedap bila diremas.', 
                     'Daun Sirsak (Annona muricata L.) merupakan pohon yang tinggi dapat mencapai sekitar 3-8 meter. Daun memanjang, bentuk lanset atau bulat telur terbalik, ujung meruncing pendek, seperti kulit, panjang 6-18 cm, tepi rata. daun yang kaya minyak dan protein serta toksisitas (Tanin, Fitat, dan Sianida) dan oleh karena itu dapat dimanfaatkan pada manusia dan hewan. Tumbuhan ini dapat tumbuh di sembarang tempat, meskipun paling baik ditanam di daerah yang cukup berair.']
        
        Manfaat = ['Kesehatan: Mengobati gatal-gatal dan luka luar', 
                   'Kesehatan: Meringankan gejala hipertensi, batuk, sariawan perut, demam, kencing manis, antipyrtik',
                   'Kesehatan: mengobati gatal-gatal dan luka luar',
                   'Kesehatan: Meredakan dan mengatasi sakit kepala, demam, kolesterol, wasir, peradangan, nyeri sendi, diabetes, keriput',
                   'Kesehatan: Perawatan kulit,demam, batuk, dan diare',
                   'Kesehatan: Mengatasi Sembelit, Gula Darah, Mencegah kanker, Meningkatkan Produksi ASI',
                   'Kesehatan: Menurunkan kadar gula darah, peradangan, mengontrol tekanan darah,  menghambat sel kanker, daya tahan tubuh',
                   'Kesehatan: pereda migrain, stres, demam, diare, sariawan, asam lambung, pereda masuk angin, antioksidan',
                   'Kesehatan: Mengontrol kadar gula darah, pilek dan flu, Meredakan Asam Urat, Antikanker dan Penangkal Tumor.', 
                   'Kesehatan: Mengatasi lemah lembung, gastritis, borok',
                   'Kesehatan: Mengatasi Sakit kepala, sembelit, kejang, kurang gizi, batuk rejan, muntah darah, kencing manis (DM), wasir, haid, penyubur rambut',
                   'Kesehatan: mengobati radang gusi, periodontitis, Jerawat,  Bisul Telinga dan kerusakan gigi',
                   'Kesehatan: Meredakan batuk, sesak napas, flu, mengobati luka, mengontrol gula darah, kesehatan gigi, bau mulut, sakit kepala, nyeri sendi',
                   'Kesehatan: Diabetes, antidiare, demam, bisul, penyakit kulit, analgesic dan imunomodulator',
                   'Kesehatan: Obat penenang, rematik, Tekanan Darah Tinggi, Kolesterol, Nafsu Makan, panu',
                   'Kesehatan: Amara, masuk angin',
                   'Kesehatan: Emetif, Obat Sariawan',
                   'Kesehatan: Antihipertensi. anti peradangan, kolesterol, tekanan darah, asam urat', 
                   'Kesehatan: Mengatasi batuk, asma, radang amandel, nafas bau, mimisan',
                   'Kesehatan: mengobati kanker, asam urat. kista, kolesterol, diabetes']
        
        Produk_Olahan = ['Ramuan tradisional: Menjemur daging buah asam jawa yang sudah dibuang kulitnya yang sudah bulatan-bulatan sekecil telur itik. Lebih jauh lagi, asam kawak ini dapat diolah menjadi madu asam, dengan cara menjemur asam kawak dalam tempat yang tertutup, hingga keluar suatu cairan cokelat kehitaman. Cairan ini madu asam digunakan untuk mengobati seriawan.', 
                         'Rebusan air daun belimbing wuluh: Siapkan 3 atau 5 lembar daun belimbing wuluh, lalu rebus dengan 2 gelas air, tunggu mendidih lalu saring pisahkan antara daun dan airnya, minum air rebusan daun belimbing wuluh selagi hangat.',
                         'Salep: Siapkan daun biduri segar yang telah dicuci bersih dan dikeringkan, Haluskan daun biduri dengan blender atau tumbuk hingga halus, Campurkan daun biduri yang telah dihaluskan dengan bahan pengikat seperti petroleum jelly atau minyak kelapa, Aduk hingga merata dan sediaan salep siap digunakan untuk mengobati luka dan gatal.',
                         'Salep: Siapkan daun cocor bebek yang telah dicuci bersih dan dikeringkan, Haluskan daun cocor bebek tersebut. Campurkan daun cocor bebek yang telah dihaluskan dengan minyak kelapa atau minyak zaitun, Panaskan campuran tersebut dengan api kecil hingga tercampur rata, Setelah campuran tersebut dingin, salep cocor bebek siap digunakan untuk membantu penyembuhan luka.',
                         'Rebusan air: Cuci bersih beberapa lembar daun jambu air, rebus hingga air mendidih, kemudian saring untuk memisahkan air rebusan dengan daun jambu air. Air rebusan daun jambu air pun siap dikonsumsi untuk mengobati diare, demam, dan sariawan, serta dapat merawat kecantikan wajah atau kulit.',
                         'Meningkatkan Produksi ASI: pilih daun katuk yang segar, cuci daun katuk supaya bersih, rebus dengan 250 ml air, tunggu hingga mendidih, setelah mendidih tuang ke gelas, kamu juga bisa menambahkan madu atau lemon untuk menambah rasa, teh daun katuk suap diminuman.',
                         'Ramuan tradisional: Daun kelor (dua genggam) dan air (dua cangkir). Rebus air sampai mendidih, masukkan daun kelor, lalu matikan api dan saring sesudah dingin. Dewasa dua kali sehari satu cangkir dan anak-anak dua kali sehari setengah cangkir.',
                         'Asam Lambung: Siapkan 3-5 lembar daun kemangi dan segelas air, masukkan air ke dalam panci dan rebus hingga mendidih, setelah itu, cemplungkan daun kemangi dan tunggu sampai warna air berubah, jika sudah mendidih dan berubah warna, segera matikan kompor, saring lembaran daun kemangi yang direbus, hanya ambil airnya saja, sebaiknya air rebusan ini diminum selagi hangat.',
                         'Obat diabetes : Cukup rebus 5 - 10 helai daun kersen yang sudah dicuci bersih ke dalam 3 gelas air. Tunggu hingga air mendidih dan tinggal tersisa sekitar 1 gelas, lalu airnya dapat langsung diminum. Ingat air rebusan daun kersen tersebut jangan ditambahi perasa lagi ya. Apalagi gula, sirup, ataupun madu.', 
                         'Demam: Lengkuas merah 1 rimpang, air hangat sedikit, madu 1 sendok makan, diparut, disaring kemudian ditambah madu, diminum 1 kali sehari 1 ramuan.',
                         'Penyubur rambut: Daun lidah buaya segar secukupnya dibelah, diambil bagian dalam yang rupanya seperti agar-agar, digosokkan ke kulit kepala sesudah mandi sore, kemudian dibungkus dengan kain, keesokan harinya rambut dicuci. Dipakai setiap hari selama 3 bulan untuk mencapai hasil yang memuaskan.',
                         'Obat Kencing Manis: cuci bersih daun mimba, didihkan air, tambahkan daun mimba ke air mendidih, rebus selama beberapa menit, saring teh dan biarkan dingin sebentar.',
                         'Asam Lambung: merebus air dan menambahkan beberapa daun mint segar ke dalamnya, biarkan teh mint direndam selama beberapa menit, lalu saring dan minum ketika hangat atau setelah makan.',
                         'Diabetes: Pilihlah sekitar 10-15 lembar daun nangka tua, siapkan dua segmen rimpang kunyit (iris tipis), siapkan tujuh gelas air putih, rebus semua bahan (angkat jika mendidih), diamkan hingga dingin (dengan wadah tertutup), konsumsi 3x sehari sebelum atau sesudah makan.',
                         'Kolesterol dan Darah Tinggi: kupas jahe dan kunyit, iris tipis. Iris tipis daun pandan, rebus air hingga mendidih, masukkan kunyit, jahe dan pandan, masak selama 2-3 menit atau hingga aromanya harum, tuang sambil saring ke gelas-gelas, tambahkan madu, aduk rata, sajikan selagi hangat, tambahkan perasan jeruk nipis secukupnya jika suka.',
                         'Demam dan mulas: Daun pepaya muda segar 1 helai, Daging buah asam secukupnya, Air 100 ml, Direbus sampai mendidih, Diminum 2 kali sehari; tiap kali minum 100 ml.',
                         'Sariawan: Daun saga 2g, daun pegagan 2g, rasuk angin 1g, kulit kayu turi 1/2 jari tangan, akar manis 1 jari tangan; air 110 ml, direbus sampai mendidih, Diminum 1 kali sehari 100 ml.',
                         'Kolesterol: Cuci bersih seledri kira-kira seberat 1 ons, panaskan 200 ml air hingga mendidih, kemudian masukkan seledri, tutup panci dan tunggu sampai 5 menit, setelah itu matikan api, saring air rebusan seledri untuk memisahkan dari ampasnya.', 
                         'Batuk: 4 lembar daun sirih. Cara membuat: direbus dengan 2 gelas air sampai mendidih. setelah dingin dipakai untuk kumur, diulang secara teratur sampai sembuh.',
                         'Resep Darah tinggi: Daun sirsak dipetik kemudian dicuci bersih lalu direbus diminum airnya, bahan dicuci bersih, dipotongpotong direbus dengan 2 gelas air jadikan 1 gelas.']
        
        Efek_Samping = ['Reaksi alergi, tekanan darah naik, gigi sensitif, dan sakit perut. sebaiknya konsultasikan ,penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi (Iritasi kulit) untuk beberapa orang sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Getahnya beracun dan dapat menyebabkan muntah sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi (Iritasi kulit) untuk beberapa orang sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Keracunan apabila dikonsumsi secara berlebihan dan membebani kerja hati sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi, sesak napas, kehilangan nafsu makan sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi sakit perut, distensi gas, diare, dan keracunan sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi dan keracunan sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi, sakit Perut, diare, mual, dan keracunan sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi', 
                        'Reaksi gatal, halusinasi dan keracunan sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi iritasi kulit, diare, menurunkan kadar gula darah dan gatal sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi, iritasi kulit,  dan keracunan sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi gatal, sesak napas, alergi, serta nyeri ulu hati. sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi, gangguan pencernaan, seperti sering buang air besar atau diare. sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi diare, mual dan keracunan sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi dan iritasi kulit sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi, diare, mual, muntah, dan kram perut, sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi alergi, diare, gangguan ginjal, dan gangguan kulit sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi iritasi gusi, alergi sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi',
                        'Reaksi mual dan muntah, diare sebaiknya konsultasikan penggunaannya dengan ahli kesehatan terlebih dahulu sebelum dikonsumsi']

        # for respond output from prediction if predict <=0.4
        if maxx <= 0.75:
            respond = jsonify({
                'message': 'Daun tidak terdeteksi'
            })
            respond.status_code = 400
            return respond

        result = {
            "Gambar": Gambar[np.argmax(pred_plant)],
            "Nama": Nama[np.argmax(pred_plant)],
            "Deskripsi": Deskripsi[np.argmax(pred_plant)],
            "Manfaat": Manfaat[np.argmax(pred_plant)],
            "Produk_Olahan": Produk_Olahan[np.argmax(pred_plant)],
            "Efek_Samping": Efek_Samping[np.argmax(pred_plant)]
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')