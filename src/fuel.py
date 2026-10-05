"""Reusable fuel-consumption training and prediction pipeline."""
from pathlib import Path
import argparse
import json
import platform
import zipfile
import urllib.request
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['cylinders', 'displacement_l', 'horsepower', 'weight_kg', 'acceleration_s', 'model_year']
FEATURES = NUMERIC + ['origin']
TARGET = 'fuel_l_per_100km'
UNITS = dict(zip(FEATURES, ['count', 'L', 'hp', 'kg', 'seconds 0-60 mph', 'year', 'usa/europe/japan']))

def load_data():
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / 'auto-mpg.data').exists():
        archive = raw / 'auto-mpg.zip'
        urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/9/auto+mpg.zip', archive)
        with zipfile.ZipFile(archive) as z:
            for name in ['auto-mpg.data', 'auto-mpg.names']:
                (raw / name).write_bytes(z.read(name))
    df = pd.read_csv(raw / 'auto-mpg.data', sep=r'\s+', names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration_s', 'model_year', 'origin', 'car_name'], na_values='?')
    if len(df) != 398 or df['mpg'].isna().any() or (df['mpg'] <= 0).any():
        raise ValueError('Unexpected Auto MPG data: expected 398 rows and positive MPG.')
    df['displacement_l'] = df['displacement'] * 0.016387064
    df['weight_kg'] = df['weight'] * 0.45359237
    df['model_year'] += 1900
    df['origin'] = df['origin'].map({1: 'usa', 2: 'europe', 3: 'japan'})
    df[TARGET] = 235.214583 / df['mpg']
    (ROOT / 'data' / 'processed').mkdir(exist_ok=True)
    df.to_csv(ROOT / 'data' / 'processed' / 'cars.csv', index=False)
    return df

def make_pipeline(model, scale=False):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['origin'])
    ])
    return Pipeline([('preprocess', preprocessor), ('model', model)])

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        'baseline': (DummyRegressor(strategy='mean'), {}, False),
        'ridge': (Ridge(), {'model__alpha': [0.1, 1, 10, 100]}, True),
        'random_forest': (RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1), {'model__max_depth': [None, 5, 10], 'model__min_samples_leaf': [1, 2, 4]}, False)
    }
    rows, searches = [], {}
    for name, (model, params, scale) in candidates.items():
        search = GridSearchCV(make_pipeline(model, scale), params, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        searches[name] = search
        rows.append({'model': name, 'cv_mae': -search.best_score_, 'cv_std': search.cv_results_['std_test_score'][search.best_index_], 'parameters': json.dumps(search.best_params_)})
    comparison = pd.DataFrame(rows).sort_values('cv_mae').reset_index(drop=True)
    name = comparison.iloc[0]['model']
    model = searches[name].best_estimator_
    predicted = model.predict(X_test)
    metrics = {'mae': mean_absolute_error(y_test, predicted), 'rmse': float(np.sqrt(mean_squared_error(y_test, predicted))), 'r2': r2_score(y_test, predicted)}
    baseline_pred = searches['baseline'].best_estimator_.predict(X_test)
    metadata = {'selected_model': name, 'features': FEATURES, 'units': UNITS, 'target': TARGET, 'train_rows': len(X_train), 'test_rows': len(X_test), 'random_state': 42, 'metrics': metrics, 'baseline_test_mae': mean_absolute_error(y_test, baseline_pred), 'cv_results': comparison.to_dict(orient='records'), 'ranges': {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERIC}, 'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'joblib']}, 'python': platform.python_version()}
    for directory in ['models', 'reports']:
        (ROOT / directory).mkdir(exist_ok=True)
    joblib.dump(model, ROOT / 'models' / 'fuel_pipeline.joblib')
    (ROOT / 'models' / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    comparison.to_csv(ROOT / 'reports' / 'model_comparison.csv', index=False)
    errors = df.loc[X_test.index, ['car_name'] + FEATURES].copy()
    errors['actual'] = y_test
    errors['predicted'] = predicted
    errors['absolute_error'] = np.abs(y_test - predicted)
    errors.sort_values('absolute_error', ascending=False).to_csv(ROOT / 'reports' / 'test_predictions.csv', index=False)
    importance = permutation_importance(model, X_train, y_train, scoring='neg_mean_absolute_error', n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({'feature': FEATURES, 'importance': importance.importances_mean, 'std': importance.importances_std}).sort_values('importance', ascending=False)
    importance_df.to_csv(ROOT / 'reports' / 'feature_importance.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, predicted, alpha=0.7)
    limits = [min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())]
    axes[0].plot(limits, limits, 'r--')
    axes[0].set(xlabel='Actual L/100 km', ylabel='Predicted L/100 km', title='Held-out test predictions')
    axes[1].hist(predicted - y_test, bins=15)
    axes[1].set(xlabel='Prediction - actual (L/100 km)', ylabel='Count', title='Residual distribution')
    fig.tight_layout()
    fig.savefig(ROOT / 'reports' / 'evaluation.png', dpi=160)
    plt.close(fig)
    report = f'''# Kết quả đánh giá\n\nMô hình chọn bằng cross-validation: **{name}**.\n\nTrain: {len(X_train)} xe; test: {len(X_test)} xe.\n\nMAE test: **{metrics['mae']:.3f} L/100 km**. RMSE: {metrics['rmse']:.3f}; R²: {metrics['r2']:.3f}.\n\nMAE baseline test: {metadata['baseline_test_mae']:.3f} L/100 km.\n\n{comparison.to_string(index=False)}\n\nPermutation importance được tính trên train, chỉ mang tính khám phá và có thể lạc quan; không chứng minh quan hệ nhân quả.\n\nDữ liệu xe 1970–1982 trong chu trình đô thị; chưa được xác thực cho xe hiện đại, xe điện hoặc chuyến đi thực tế. Chia ngẫu nhiên đánh giá xe trong cùng phân phối lịch sử, không đánh giá khả năng dự đoán tương lai.\n'''
    (ROOT / 'reports' / 'results.md').write_text(report, encoding='utf-8')
    return {'model': model, 'metadata': metadata, 'comparison': comparison, 'errors': errors, 'importance': importance_df, 'X_train': X_train, 'X_test': X_test}

def validate_vehicle(vehicle):
    if not isinstance(vehicle, dict):
        raise ValueError('Input must be a JSON object.')
    missing = set(FEATURES) - {'horsepower'} - vehicle.keys()
    if missing:
        raise ValueError(f'Missing fields: {sorted(missing)}')
    extra = vehicle.keys() - set(FEATURES)
    if extra:
        raise ValueError(f'Unknown fields: {sorted(extra)}')
    row = {c: vehicle.get(c) for c in FEATURES}
    for c in NUMERIC:
        value = row[c]
        if c == 'horsepower' and value is None:
            row[c] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{c} must be a finite positive number.')
        if c in ['cylinders', 'model_year'] and value != int(value):
            raise ValueError(f'{c} must be an integer.')
        if c == 'model_year' and not 1886 <= value <= 2100:
            raise ValueError('model_year must be a full year between 1886 and 2100.')
    if not isinstance(row['origin'], str) or row['origin'] not in ['usa', 'europe', 'japan']:
        raise ValueError('origin must be usa, europe or japan.')
    return pd.DataFrame([row], columns=FEATURES)

def predict(vehicle):
    frame = validate_vehicle(vehicle)
    model = joblib.load(ROOT / 'models' / 'fuel_pipeline.joblib')
    metadata = json.loads((ROOT / 'models' / 'metadata.json').read_text(encoding='utf-8'))
    warnings = []
    for c, (low, high) in metadata['ranges'].items():
        value = frame.iloc[0][c]
        if pd.notna(value) and not low <= value <= high:
            warnings.append(f'{c}={value} outside training range [{low}, {high}]; extrapolation.')
    value = float(model.predict(frame)[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Model produced a nonphysical prediction; review the input.')
    return {'fuel_l_per_100km': value, 'model': metadata['selected_model'], 'warnings': warnings, 'context': 'Historical urban-cycle data; not validated for modern vehicles.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'train':
            print(json.dumps(train()['metadata']['metrics'], indent=2))
        else:
            if args.input is None:
                parser.error('predict requires --input vehicle.json')
            print(json.dumps(predict(json.loads(args.input.read_text(encoding='utf-8-sig'))), indent=2, ensure_ascii=False))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, f'Error: {exc}\n')

if __name__ == '__main__':
    main()
