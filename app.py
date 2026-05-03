from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

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
def preprocess_and_predict(raw_input: dict) -> float:
    df = pd.DataFrame([raw_input])
    print("=== KOLOM DF ===", df.columns.tolist())

    # 1. Normalisasi string kategorikal
    for col in OHE_COLS:
        if col in df.columns:
            print(f"Normalize OHE [{col}]:", df[col].tolist())
            df[col] = df[col].apply(lambda v: normalize_input(str(v), OHE_CATEGORIES[col]))

    for col in ORD_COLS:
        if col in df.columns:
            print(f"Normalize ORD [{col}]:", df[col].tolist())
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
    print("=== KOLOM SEBELUM REORDER ===", df.columns.tolist())
    for col in feature_names:
        if col not in df.columns:
            print(f"KOLOM HILANG, set 0: {col}")
            df[col] = 0
    df = df[feature_names]

    # 8. StandardScaler
    df[num_scale_cols] = scaler.transform(df[num_scale_cols])

    # 9. Predict
    print("=== FINAL FEATURES ===", df.to_dict())
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
        print("=== DATA MASUK ===", data)

        # ── Default field kategorikal ──
        data.setdefault('gender', 'Male')
        data.setdefault('region', 'Asia')
        data.setdefault('income_level', 'Upper-Mid')
        data.setdefault('education_level', 'Bachelor')
        data.setdefault('daily_role', 'Student')

        # ── Fix device_type ──
        device_type_map = {
            'web': 'Laptop',
            'android': 'Android',
            'iphone': 'iPhone',
            'tablet': 'Tablet',
        }
        if 'device_type' in data:
            data['device_type'] = device_type_map.get(
                str(data['device_type']).strip().lower(),
                'Laptop'
            )

        # ── Fix gender (kapitalisasi) ──
        gender_map = {'male': 'Male', 'female': 'Female'}
        if 'gender' in data:
            data['gender'] = gender_map.get(
                str(data['gender']).strip().lower(), 'Male'
            )

        # ── Fix education_level ──
        education_map = {
            'high school': 'High School',
            'sma':         'High School',
            'diploma':     'Bachelor',
            'd3':          'Bachelor',
            'd4':          'Bachelor',
            'bachelor':    'Bachelor',
            's1':          'Bachelor',
            'master':      'Master',
            's2':          'Master',
            'phd':         'PhD',
            's3':          'PhD',
        }
        if 'education_level' in data:
            data['education_level'] = education_map.get(
                str(data['education_level']).strip().lower(), 'Bachelor'
            )

        # ── Hapus field yang tidak dipakai model ──
        fields_to_remove = [
            'questionnaire_id',
            'date_of_birth',
            'age',
            'study_minutes',
            'physical_activity_days',
        ]
        for field in fields_to_remove:
            data.pop(field, None)

        print("=== DATA FINAL ===", data)

        hasil = preprocess_and_predict(data)

        return jsonify({
            'digital_dependence_score': hasil,
            'category':                get_category(hasil),
            'confidence':              1.0,
            'status':                  'ok',
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