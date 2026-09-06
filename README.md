# 🏡 California House Price Predictor — App

This is a **Streamlit web app** that uses the final tuned model
(`HistGradientBoostingRegressor`) from your notebook (`15_4_house_price_prediction.ipynb`)
to predict house prices, wrapped in a clean and interactive UI.

## 📦 Files
- `app.py` — Main Streamlit app (UI + prediction logic)
- `house_price_model.pkl` — Pre-trained pipeline (preprocessing + model), same as the notebook
- `model_meta.json` — Metadata for slider ranges, categories, and test metrics
- `housing.csv` — Dataset (for reference only, the app doesn't need it to run)
- `requirements.txt` — Required Python packages

## ▶️ How to run

1. Keep all files in one folder (already done here).
2. Open a terminal inside this folder:
   ```bash
   cd house_app
   ```
3. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. The browser should open automatically — if not, open the
   `http://localhost:8501` link shown in the terminal.

## 🖱️ How to use the app

- In the left sidebar, enter location (latitude/longitude), ocean proximity,
  house age, rooms, bedrooms, households, population, and income.
- Click **"🔮 Predict House Price"**.
- On the right side you'll see the estimated price, an error range, and a map
  plotting the location you entered.

## 🔁 Retraining the model

If you update the dataset or want to retrain the model, use the same
preprocessing pipeline (median imputer + StandardScaler for numeric features,
most-frequent imputer + OneHotEncoder for `ocean_proximity`) and
`HistGradientBoostingRegressor` (best params: `l2_regularization=0.1,
learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20`) to produce a new
`house_price_model.pkl` and replace the old one — the app will automatically
load the new model.

## ✨ Deploying (optional)

If you want a live app on the internet:
- You can deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud) —
  just push these files to a GitHub repo and click "Deploy".
