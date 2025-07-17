# 🗺️ Destinations App – What Are You Thinking?

**Destinations App** este o aplicație interactivă construită cu Streamlit care oferă trei funcționalități captivante:

- 🧠 „What am I thinking?” – Sistem care ghicește o destinație turistică pe baza întrebărilor despre atributele sale
- ❓ Quiz – Testează-ți cunoștințele despre o anumită destinație
- 🌍 Recomandări – Primește sugestii personalizate de locuri în care să călătorești, folosind K-Means clustering

---

## 🚀 Funcționalități principale

- 🌐 Interfață prietenoasă cu utilizatorul (Streamlit)
- 📊 Procesare și curățare automată a datelor despre destinații
- 🧩 Întrebări generate automat din atributele destinațiilor
- 🤖 Recomandări bazate pe algoritmi de machine learning (scikit-learn)
- 🇷🇴 Aplicație 100% în limba română

---

## 🛠️ Tehnologii folosite

- **Python 3.11+**
- **Streamlit** pentru interfață web
- **Pandas**, **NumPy** pentru procesare date
- **scikit-learn** pentru clustering și clasificare
- **Tkinter** (pentru alte interfețe grafice – opțional)

---

## 🔧 Instalare locală

1. Clonează repository-ul:
   ```bash
   git clone https://github.com/username/destinations-app.git
   cd destinations-app
   ```
2. Creează un mediu virtual (opțional dar recomandat):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

3. Instalează dependințele:
   ```bash
   pip install -r requirements.txt

4. Rulează aplicația:
   ```bash
   streamlit run app.py

5. Vizitează http://localhost:8501 în browser dacă nu se deschide automat.
