import sys
import unittest
from pathlib import Path
import json
import joblib
import numpy as np
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import fuel

class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.vehicle = json.loads((fuel.ROOT / 'examples' / 'vehicle.json').read_text())

    def test_data_and_conversions(self):
        df = fuel.load_data()
        self.assertEqual(len(df), 398)
        self.assertEqual(df.horsepower.isna().sum(), 6)
        self.assertAlmostEqual(df.iloc[0].displacement_l, 307 * 0.016387064)
        self.assertAlmostEqual(df.iloc[0].weight_kg, 3504 * 0.45359237)
        self.assertAlmostEqual(df.iloc[0][fuel.TARGET], 235.214583 / 18)
        self.assertEqual(set(df.origin), {'usa', 'europe', 'japan'})

    def test_prediction_and_reload(self):
        result = fuel.predict(self.vehicle)
        self.assertGreater(result['fuel_l_per_100km'], 0)
        path = fuel.ROOT / 'models' / 'fuel_pipeline.joblib'
        X = fuel.validate_vehicle(self.vehicle)
        model = joblib.load(path)
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'pipeline.joblib'
            joblib.dump(model, saved)
            np.testing.assert_allclose(model.predict(X), joblib.load(saved).predict(X))

    def test_missing_power(self):
        self.vehicle.pop('horsepower')
        self.assertGreater(fuel.predict(self.vehicle)['fuel_l_per_100km'], 0)
        self.vehicle['horsepower'] = None
        self.assertTrue(fuel.validate_vehicle(self.vehicle).horsepower.isna().all())

    def test_invalid_inputs(self):
        for key, value in [('weight_kg', -1), ('cylinders', True), ('origin', []), ('model_year', 80), ('horsepower', float('inf'))]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    fuel.validate_vehicle({**self.vehicle, key: value})
        self.vehicle.pop('weight_kg')
        with self.assertRaises(ValueError):
            fuel.validate_vehicle(self.vehicle)

    def test_extrapolation(self):
        self.vehicle['model_year'] = 2025
        self.assertTrue(fuel.predict(self.vehicle)['warnings'])

if __name__ == '__main__':
    unittest.main()
