import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 0) App Configuration
st.set_page_config(page_title="Weather Forecast App", layout="wide")
st.title("🌤️ Weather Prediction")

# 1) Get list of provinces from CSV folder
def get_province_list(data_folder):
    try:
        files = os.listdir(data_folder)
    except FileNotFoundError:
        st.error(f"Folder not found: {data_folder}")
        return []
    return sorted([os.path.splitext(f)[0] for f in files if f.lower().endswith(".csv")])

# 2) Load CSV file
@st.cache_data
def load_file(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["snow", "wdir", "wpgt", "tsun"] if c in df.columns])
    df = df.dropna(subset=["tavg", "tmin", "tmax", "wspd", "pres", "prcp"])
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()

    # Label for tomorrow
    df['label'] = df['prcp'].apply(lambda x: "Rain" if x > 0 else "Sunny")
    df['label_next'] = df['label'].shift(-1)

    # Use rain_lag_1
    df['rain_flag'] = df['prcp'].gt(0).astype(int)
    df['rain_lag_1'] = df['rain_flag'].shift(1)
    df = df.dropna(subset=["label_next", "rain_lag_1"]).reset_index(drop=True)

    # Balance the dataset
    rain_df = df[df['label_next'] == "Rain"]
    sunny_df = df[df['label_next'] == "Sunny"]
    if len(rain_df) > 0 and len(sunny_df) > 0:
        majority = sunny_df if len(sunny_df) > len(rain_df) else rain_df
        minority = rain_df if len(rain_df) < len(sunny_df) else sunny_df
        minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        df = pd.concat([majority, minority_up]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df, last_date

# 3) Forecasting
def forecast(df, last_date, n_days):
    feat_clf = ["tavg", "tmin", "tmax", "wspd", "pres", "rain_lag_1"]
    feat_reg = ["tavg", "tmin", "tmax", "wspd", "pres", "prcp", "rain_lag_1"]
    reg_targets = ["tmax", "tmin", "prcp", "pres", "wspd"]

    # Train-test split for Naive Bayes
    X_clf = df[feat_clf]
    y_clf = df["label_next"]
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    
    # Normalize features for Naive Bayes
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clf)
    X_test_scaled  = scaler.transform(X_test_clf)

    # Naive Bayes model
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train_clf)
    y_pred_clf = nb.predict(X_test_scaled)
    acc = accuracy_score(y_test_clf, y_pred_clf)

    # Train-test split for Random Forest Regression
    X_reg = df[feat_reg]
    y_regs = {t: df[t] for t in reg_targets}
    X_train_reg, X_test_reg = train_test_split(X_reg, test_size=0.2, random_state=42)
    rf_models = {}
    reg_scores = []
    for tgt in reg_targets:
        y_tr = y_regs[tgt].loc[X_train_reg.index]
        y_te = y_regs[tgt].loc[X_test_reg.index]
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf.fit(X_train_reg, y_tr)
        y_pr = rf.predict(X_test_reg)
        rf_models[tgt] = rf
        reg_scores.append({
            "Target": tgt,
            "MAE": mean_absolute_error(y_te, y_pr),
            "MSE": mean_squared_error(y_te, y_pr),
            "R²": r2_score(y_te, y_pr)
        })

    # Initialize for forecasting
    last = df.iloc[-1]
    rows = []
    classes = list(nb.classes_)

    for i in range(1, n_days+1):
        clf_feat = np.array([[last['tavg'], last['tmin'], last['tmax'], last['wspd'], last['pres'], last['rain_lag_1']]])
        clf_scaled = scaler.transform(clf_feat)
        prob = nb.predict_proba(clf_scaled)[0]
        pred_lbl = "Rain" if prob[classes.index("Rain")] > 0.5 else "Sunny"
        rain_prob = prob[classes.index("Rain")]

        reg_feat = np.array([[last['tavg'], last['tmin'], last['tmax'], last['wspd'], last['pres'], last['prcp'], last['rain_lag_1']]])
        preds = {t: rf_models[t].predict(reg_feat)[0] for t in reg_targets}

        rows.append({
            "Date": (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "Forecast": pred_lbl,
            "Rain Probability": f"{rain_prob*100:.1f}%",
            "Tmax (°C)": f"{preds['tmax']:.1f}",
            "Tmin (°C)": f"{preds['tmin']:.1f}",
            "Precipitation (mm)": f"{preds['prcp']:.1f}",
            "Wind Speed (m/s)": f"{preds['wspd']:.1f}",
            "Pressure (hPa)": f"{preds['pres']:.1f}"
        })

        # Update for next prediction
        new_rain = 1 if preds['prcp'] > 0 else 0
        last = pd.Series({
            'tavg': (preds['tmax'] + preds['tmin']) / 2,
            'tmin': preds['tmin'],
            'tmax': preds['tmax'],
            'wspd': preds['wspd'],
            'pres': preds['pres'],
            'prcp': preds['prcp'],
            'rain_lag_1': new_rain
        })

    return pd.DataFrame(rows), acc, y_test_clf, y_pred_clf, reg_scores

# 4) UI
folder = "data"
provinces = get_province_list(folder)
col1, col2 = st.columns([2, 5])
with col1:
    st.subheader("🔧 Settings")
    sel = st.selectbox("📍 Province", provinces)
    days = st.number_input("📆 Days", 1, 7, 3)
    run = st.button("🔍 Run Forecast")
with col2:
    st.subheader("📋 Results")
    if not run:
        st.info("Select settings and click Run")
    else:
        df, last_date = load_file(os.path.join(folder, f"{sel}.csv"))
        res, acc, y_test, y_pred, scores = forecast(df, last_date, days)
        st.markdown("### 📈 Forecast")
        st.dataframe(res)
        st.markdown(f"**Accuracy:** {acc*100:.2f}%")
        st.markdown("### 📊 Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
        st.markdown("### 📊 Regression Errors")
        st.dataframe(pd.DataFrame(scores))
