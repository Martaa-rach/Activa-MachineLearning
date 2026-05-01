from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# ============================================================
# LOAD MODEL — dijalankan SEKALI saat Flask start
# ============================================================
with open('digital_dependence_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

model              = pkg['model']
scaler             = pkg['scaler']
ohe                = pkg['ohe']
ord_enc            = pkg['ord_enc']
log_cols           = pkg['log_cols']            # ['notifications_per_day', 'device_hours_per_day']
sqrt_cols          = pkg['sqrt_cols']           # ['social_media_mins', 'anxiety_score', 'phone_unlocks']
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
# HELPER — normalisasi string dari user (case-insensitive)
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
# HELPER — preprocessing + prediksi  ← TARUH DI SINI
# ============================================================
def preprocess_and_predict(raw_input: dict) -> float:
    df = pd.DataFrame([raw_input])

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

    # 5. Transform log1p dan sqrt — KUNCI AGAR TIDAK MELEDAK
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    for col in sqrt_cols:
        if col in df.columns:
            df[col] = np.sqrt(df[col])

    # 6. Drop kolom multikolinear & tidak signifikan
    df = df.drop(columns=drop_multicolinear + drop_insig, errors='ignore')

    # 7. Pastikan semua kolom ada & urutkan sesuai training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # 8. StandardScaler
    df[num_scale_cols] = scaler.transform(df[num_scale_cols])

    # 9. Predict
    result = model.predict(df)[0]
    return round(float(result), 2)

def get_category(score: float) -> str:
    if score < 40:
        return 'rendah'
    elif score < 70:
        return 'sedang'
    else:
        return 'tinggi'


# ============================================================
# ROUTES
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ── Fix device_type: "Web" tidak dikenal model ──
        device_type_map = {
            'web': 'Laptop',
            'android': 'Android',
            'iphone': 'iPhone',
            'tablet': 'Tablet',
        }
        if 'device_type' in data:
            data['device_type'] = device_type_map.get(
                str(data['device_type']).strip().lower(),
                'Laptop'  # default fallback
            )

        # Hapus field yang tidak dikenal model (dikirim Laravel tapi tidak dipakai)
        fields_to_remove = [
            'questionnaire_id',
            'date_of_birth',
            'study_minutes',
            'physical_activity_days',
            'sleep_hours',
            'sleep_quality',
            'depression_score',
            'stress_level',
            'happiness_score',
        ]
        for field in fields_to_remove:
            data.pop(field, None)

        hasil = preprocess_and_predict(data)

        return jsonify({
            'digital_dependence_score' : hasil,
            'category'                 : get_category(hasil),  # ✅ tambah
            'confidence'               : 1.0,                  # ✅ tambah
            'status'                   : 'ok'
        })

    except ValueError as ve:
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