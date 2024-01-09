import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model
model_klasifikasi = pickle.load(open('model_klasifikasi_svm.sav', 'rb'))

# Instansiasi TfidfVectorizer
tfidf = TfidfVectorizer()

# Load vocabulary from saved file
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

# Judul halaman web
st.title('Prediksi Topik Bidang IT yang Dibahas')

clean_teks = st.text_input('Masukkan Teks')

topik_detection = ''

if st.button('Hasil Deteksi Topik '):
    # Ubah data uji menjadi list untuk fit_transform
    clean_teks_list = [clean_teks]
    
    # Melatih (fit) vektorizer dengan data pelatihan (pastikan Anda memilikinya)
    # Jika sudah dilatih sebelumnya, Anda dapat melewatkan langkah ini
    loaded_vec.fit_transform(clean_teks_list)
    
    # Transform data uji menggunakan transform (bukan fit_transform)
    transformed_text = loaded_vec.transform(clean_teks_list)
    
    # Ubah data sparse menjadi dense
    dense_transformed_text = transformed_text.toarray()
    
    # Lakukan prediksi
    predict_topik = model_klasifikasi.predict(dense_transformed_text)
    
    if (predict_topik == 1):
        topik_detection = 'Jaringan'
    elif(predict_topik == 2):
        topik_detection = 'Perangkat Lunak'
    else:
        topik_detection = 'Sistem Cerdas'

st.success(topik_detection)
