from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
import os
from scipy.spatial.distance import mahalanobis

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================
with open('digital_dependence_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

model              = pkg['model']
scaler             = pkg['scaler']
ohe                = pkg['ohe']
ord_enc            = pkg['ord_enc']
log_cols           = pkg['log_cols']
sqrt_cols          = pkg['sqrt_cols']
winsor_bounds      = pkg['winsor_bounds']
num_scale_cols     = pkg['num_scale_cols']
feature_names      = pkg['feature_names']
drop_multicolinear = pkg['drop_multicolinear']
drop_insig         = pkg['drop_insig']

OHE_COLS = ['gender', 'region', 'device_type', 'income_level']
ORD_COLS = ['education_level', 'daily_role']

OHE_CATEGORIES = {col: list(cats) for col, cats in zip(OHE_COLS, ohe.categories_)}
ORD_CATEGORIES = {col: list(cats) for col, cats in zip(ORD_COLS, ord_enc.categories_)}

# ============================================================
# CONFIDENCE SETUP
# Setelah StandardScaler, distribusi fitur ≈ mean=0, std=1
# Sehingga: mean vektor = [0, 0, ...], cov = identity matrix
# ============================================================
SCORE_MIN = 0.0
SCORE_MAX = 100.0

_feature_mean    = np.zeros(len(feature_names))
_feature_cov_inv = np.eye(len(feature_names))  # inverse of identity = identity


# ============================================================
# CONFIDENCE FUNCTIONS
# ============================================================
def confidence_by_score(score: float) -> dict:
    """
    Confidence berbasis posisi score dalam range 0–100.
    Score mendekati ekstrem (0 atau 100) → confidence turun.
    Range output: 60–100%.
    """
    score_clipped  = float(np.clip(score, SCORE_MIN, SCORE_MAX))
    midpoint       = (SCORE_MIN + SCORE_MAX) / 2  # 50.0
    distance_ratio = abs(score_clipped - midpoint) / midpoint  # 0.0–1.0
    confidence     = 1.0 - (distance_ratio * 0.4)  # turun maks 40%
    confidence_pct = round(float(np.clip(confidence, 0.6, 1.0)) * 100, 2)

    if score_clipped <= 25:
        label = 'Rendah'
    elif score_clipped <= 50:
        label = 'Sedang'
    elif score_clipped <= 75:
        label = 'Tinggi'
    else:
        label = 'Sangat Tinggi'

    return {'confidence_pct': confidence_pct, 'label': label}


def confidence_by_distance(X_scaled: np.ndarray) -> dict:
    """
    Confidence berbasis Mahalanobis distance dari distribusi training.
    Input jauh dari distribusi → confidence turun.
    """
    dist           = mahalanobis(X_scaled[0], _feature_mean, _feature_cov_inv)
    confidence     = 1.0 / (1.0 + (dist / 3.0))
    confidence_pct = round(float(np.clip(confidence, 0.0, 1.0)) * 100, 2)

    if dist <= 1.0:
        zone = 'Dalam distribusi normal'
    elif dist <= 2.0:
        zone = 'Agak di luar distribusi'
    elif dist <= 3.0:
        zone = 'Jauh dari distribusi'
    else:
        zone = 'Outlier — prediksi kurang andal'

    return {
        'confidence_pct'       : confidence_pct,
        'mahalanobis_distance' : round(float(dist), 4),
        'zone'                 : zone,
    }


def get_combined_confidence(score: float, X_scaled: np.ndarray) -> dict:
    """
    Gabungkan kedua pendekatan dengan bobot 50:50.
    """
    conf_score = confidence_by_score(score)
    conf_dist  = confidence_by_distance(X_scaled)

    combined = round(
        0.5 * conf_score['confidence_pct'] +
        0.5 * conf_dist['confidence_pct'],
        2
    )

    return {
        'confidence_final_pct'    : combined,
        'label'                   : conf_score['label'],
        'confidence_by_score_pct' : conf_score['confidence_pct'],
        'confidence_by_distance'  : conf_dist,
    }


# ============================================================
# HELPER — normalisasi string
# ============================================================
def normalize_input(value: str, valid_categories: list) -> str:
    if value in valid_categories:
        return value
    value_lower = value.strip().lower()
    for cat in valid_categories:
        if cat.lower() == value_lower:
            return cat
    raise ValueError(f"Nilai '{value}' tidak valid. Pilihan: {valid_categories}")


# ============================================================
# HELPER — preprocessing + prediksi
# ============================================================
def preprocess_and_predict(raw_input: dict):
    """
    Returns tuple: (score: float, X_scaled: np.ndarray)
    X_scaled dibutuhkan untuk menghitung Mahalanobis distance.
    """
    df = pd.DataFrame([raw_input])
    print("=== KOLOM DF ===", df.columns.tolist())

    # 1. Normalisasi string kategorikal
    for col in OHE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: normalize_input(str(v), OHE_CATEGORIES[col]))

    for col in ORD_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: normalize_input(str(v), ORD_CATEGORIES[col]))

    # 2. Ordinal Encoding
    df[ORD_COLS] = ord_enc.transform(df[ORD_COLS])

    # 3. One-Hot Encoding
    ohe_arr = ohe.transform(df[OHE_COLS])
    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(OHE_COLS))
    df = pd.concat([df.drop(columns=OHE_COLS).reset_index(drop=True), ohe_df], axis=1)

    # 4. Winsorize
    for col, (lo, hi) in winsor_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # 5. Transform log1p dan sqrt
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    for col in sqrt_cols:
        if col in df.columns:
            df[col] = np.sqrt(df[col])

    # 6. Drop kolom multikolinear & tidak signifikan
    df = df.drop(columns=drop_multicolinear + drop_insig, errors='ignore')

    # 7. Pastikan semua kolom ada & urutkan
    for col in feature_names:
        if col not in df.columns:
            print(f"KOLOM HILANG, set 0: {col}")
            df[col] = 0
    df = df[feature_names]

    # 8. StandardScaler
    df[num_scale_cols] = scaler.transform(df[num_scale_cols])

    # 9. Simpan X_scaled untuk confidence, lalu predict
    X_scaled = df[feature_names].values
    score    = round(float(model.predict(df)[0]), 2)

    return score, X_scaled


def get_category(score: float) -> str:
    if score < 33.47:
        return 'rendah'
    elif score < 61.34:
        return 'sedang'
    else:
        return 'tinggi'



# ============================================================
# RULE — tentukan penyebab dari SHAP + raw input
# ============================================================
def apply_rules(raw_data: dict) -> list:
    penyebab = []

    if raw_data.get("device_hours_per_day", 0) >= 9.15:
        penyebab.append("screen_time_high")

    if raw_data.get("notifications_per_day", 0) >= 434:
        penyebab.append("notification_overload")

    if raw_data.get("sleep_hours", 0) < 6.41:
        penyebab.append("sleep_low")

    if raw_data.get("sleep_quality", 0) <= 1.92:
        penyebab.append("sleep_bad_quality")

    if raw_data.get("anxiety_score", 0) >= 8.85:
        penyebab.append("anxiety_high")

    if raw_data.get("depression_score", 0) >= 13:
        penyebab.append("depression_high")

    if raw_data.get("stress_level", 0) >= 8.79:
        penyebab.append("stress_high")

    if raw_data.get("happiness_score", 0) <= 4:
        penyebab.append("happiness_low")

    if not penyebab:
        penyebab.append("general")

    return penyebab

# ============================================================
# ROUTES
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("=== DATA MASUK ===", data)

        # ── Default field kategorikal ──
        data.setdefault('gender', 'Male')
        data.setdefault('region', 'Asia')
        data.setdefault('income_level', 'Upper-Mid')
        data.setdefault('education_level', 'Bachelor')
        data.setdefault('daily_role', 'Student')

        # ── Fix device_type ──
        device_type_map = {
            'web'    : 'Laptop',
            'android': 'Android',
            'iphone' : 'iPhone',
            'tablet' : 'Tablet',
        }
        if 'device_type' in data:
            data['device_type'] = device_type_map.get(
                str(data['device_type']).strip().lower(), 'Laptop'
            )

        # ── Fix gender ──
        gender_map = {'male': 'Male', 'female': 'Female'}
        if 'gender' in data:
            data['gender'] = gender_map.get(
                str(data['gender']).strip().lower(), 'Male'
            )

        # ── Fix education_level ──
        education_map = {
            'high school': 'High School',
            'sma'        : 'High School',
            'diploma'    : 'Bachelor',
            'd3'         : 'Bachelor',
            'd4'         : 'Bachelor',
            'bachelor'   : 'Bachelor',
            's1'         : 'Bachelor',
            'master'     : 'Master',
            's2'         : 'Master',
            'phd'        : 'PhD',
            's3'         : 'PhD',
        }
        if 'education_level' in data:
            data['education_level'] = education_map.get(
                str(data['education_level']).strip().lower(), 'Bachelor'
            )

        # ── Hapus field yang tidak dipakai model ──
        for field in ['questionnaire_id', 'date_of_birth', 'age',
                      'study_minutes', 'physical_activity_days']:
            data.pop(field, None)

        print("=== DATA FINAL ===", data)

        # ── Preprocess & Predict ──
        score, X_scaled = preprocess_and_predict(data)

        # ── Hitung Confidence ──
        confidence = get_combined_confidence(score, X_scaled)

        # ── RULE → penyebab ──
        penyebab = apply_rules(data)
        print("PENYEBAB:", penyebab)

        # ── Kirim ke Node.js AI ──
        node_payload = {
            "score"    : score,
            "category" : get_category(score),
            "penyebab" : penyebab,   # hasil SHAP + RULE
            "data"     : data        # raw input untuk konteks AI
        }

        NODE_URL = os.getenv("NODE_CHATBOT_URL", "http://localhost:3000")

        try:
            ai_resp   = requests.post(f"{NODE_URL}/chatbot", json=node_payload, timeout=30)
            ai_result = ai_resp.json()
        except Exception as e:
            ai_result = {"error": f"Node.js tidak merespons: {str(e)}"}

        # ── Susun ai_analysis dari hasil Node.js ──
        ai_data = ai_result.get("ai", {})
        ai_analysis = {
            "penyebab"     : ai_data.get("penyebab", penyebab),
            "pembukaan"    : ai_data.get("pembukaan", ""),
            "rekomendasi"  : ai_data.get("rekomendasi", []),
            "generated_at" : __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }

        # ── Return ke Laravel → Flutter ──
        return jsonify({
            "digital_dependence_score" : score,
            "category"                 : get_category(score),
            "confidence"               : confidence,
            "high_risk_flag"           : 1 if score >= 70 else 0,
            "ai_analysis"              : ai_analysis,
            "status"                   : "ok",
        })

    except ValueError as ve:
        print("=== VALUE ERROR ===", str(ve))
        return jsonify({'error': str(ve), 'status': 'error'}), 422
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 400


@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'feature_names'  : feature_names,
        'ohe_categories' : OHE_CATEGORIES,
        'ord_categories' : ORD_CATEGORIES,
        'log_cols'       : log_cols,
        'sqrt_cols'      : sqrt_cols,
        'winsor_bounds'  : {k: list(v) for k, v in winsor_bounds.items()},
        'scaler_cols'    : scaler.feature_names_in_.tolist(),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)