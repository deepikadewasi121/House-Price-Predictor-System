# 🏡 California House Price Predictor — App

Ye ek **Streamlit web app** hai jo aapke notebook (`15_4_house_price_prediction.ipynb`) ke
final tuned model (`HistGradientBoostingRegressor`) ko use karke house price predict karta hai,
ek clean aur interactive UI ke saath.

## 📦 Files
- `app.py` — Main Streamlit app (UI + prediction logic)
- `house_price_model.pkl` — Pre-trained pipeline (preprocessing + model), notebook jaisa hi
- `model_meta.json` — Slider ranges, categories aur test metrics ke liye metadata
- `housing.csv` — Dataset (sirf reference ke liye, app ko iski zaroorat nahi)
- `requirements.txt` — Required Python packages

## ▶️ Kaise chalayein (How to run)

1. Sab files ek folder me rakho (already done here).
2. Terminal me is folder ke andar jao:
   ```bash
   cd house_app
   ```
3. (Recommended) Ek virtual environment bana lo:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
4. Dependencies install karo:
   ```bash
   pip install -r requirements.txt
   ```
5. App run karo:
   ```bash
   streamlit run app.py
   ```
6. Browser me automatically khul jayega — agar nahi khule to terminal me diya gaya
   `http://localhost:8501` link open kar lo.

## 🖱️ App kaise use karein

- Left sidebar me location (latitude/longitude), ocean proximity, house age, rooms,
  bedrooms, households, population, aur income daalo.
- **"🔮 Predict House Price"** button dabao.
- Right side me estimated price, error range, aur ek map dikhega jisme aapki di hui
  location plot hogi.

## 🔁 Model ko retrain karna ho to

Agar aap dataset update karte ho ya model ko dobara train karna chahte ho, to isi
preprocessing pipeline (median imputer + StandardScaler for numeric, most-frequent
imputer + OneHotEncoder for `ocean_proximity`) aur `HistGradientBoostingRegressor`
(best params: `l2_regularization=0.1, learning_rate=0.1, max_leaf_nodes=63,
min_samples_leaf=20`) ka use karke naya `house_price_model.pkl` bana ke replace kar dena
— app automatically naya model load kar lega.

## ✨ Deploy karna ho (optional)

Agar internet par live app chahiye:
- [Streamlit Community Cloud](https://streamlit.io/cloud) par free me deploy ho sakta hai —
  bas GitHub repo banao (ye saari files daal ke) aur "Deploy" button dabao.
