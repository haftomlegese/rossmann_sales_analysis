import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

class TestEDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load datasets
        cls.train = pd.read_csv('../data/train.csv', low_memory=False)
        cls.test = pd.read_csv('../data/test.csv', low_memory=False)
        cls.store = pd.read_csv('../data/store.csv', low_memory=False)
        
        # Merge store information with train and test datasets
        cls.train = pd.merge(cls.train, cls.store, on='Store', how='left')
        cls.test = pd.merge(cls.test, cls.store, on='Store', how='left')

    def test_data_loading(self):
        self.assertFalse(self.train.empty, "Train dataset is empty")
        self.assertFalse(self.test.empty, "Test dataset is empty")
        self.assertIn('Sales', self.train.columns, "Sales column is missing from train dataset")

    def test_missing_values_handling(self):
        # Fill missing values
        self.train['CompetitionDistance'].fillna(self.train['CompetitionDistance'].median(), inplace=True)
        self.train['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        self.train['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        self.train['Promo2SinceWeek'].fillna(0, inplace=True)
        self.train['Promo2SinceYear'].fillna(0, inplace=True)
        self.train['PromoInterval'].fillna(0, inplace=True)

        self.test['CompetitionDistance'].fillna(self.test['CompetitionDistance'].median(), inplace=True)
        self.test['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        self.test['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        self.test['Promo2SinceWeek'].fillna(0, inplace=True)
        self.test['Promo2SinceYear'].fillna(0, inplace=True)
        self.test['PromoInterval'].fillna(0, inplace=True)
        self.test['Open'].fillna(1, inplace=True)

        # Check for any remaining missing values
        self.assertEqual(self.train.isnull().sum().sum(), 0, "Train dataset still has missing values")
        self.assertEqual(self.test.isnull().sum().sum(), 0, "Test dataset still has missing values")

    def test_conversion_to_numeric(self):
        # Convert categorical variables to numeric
        self.train['StateHoliday'] = self.train['StateHoliday'].map({'0': 0, 'a': 1, 'b': 2, 'c': 3})
        self.test['StateHoliday'] = self.test['StateHoliday'].map({'0': 0, 'a': 1, 'b': 2, 'c': 3})

        self.train['StoreType'] = self.train['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
        self.test['StoreType'] = self.test['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})

        self.train['Assortment'] = self.train['Assortment'].map({'a': 1, 'b': 2, 'c': 3})
        self.test['Assortment'] = self.test['Assortment'].map({'a': 1, 'b': 2, 'c': 3})

        self.assertTrue(self.train['StateHoliday'].dtype == np.int64, "StateHoliday column is not numeric in train dataset")
        self.assertTrue(self.test['StateHoliday'].dtype == np.int64, "StateHoliday column is not numeric in test dataset")
        self.assertTrue(self.train['StoreType'].dtype == np.int64, "StoreType column is not numeric in train dataset")
        self.assertTrue(self.test['StoreType'].dtype == np.int64, "StoreType column is not numeric in test dataset")
        self.assertTrue(self.train['Assortment'].dtype == np.int64, "Assortment column is not numeric in train dataset")
        self.assertTrue(self.test['Assortment'].dtype == np.int64, "Assortment column is not numeric in test dataset")

    def test_feature_engineering(self):
        # Extract datetime features
        self.train['Date'] = pd.to_datetime(self.train['Date'])
        self.train['Year'] = self.train['Date'].dt.year
        self.train['Month'] = self.train['Date'].dt.month
        self.train['Day'] = self.train['Date'].dt.day
        self.train['DayOfWeek'] = self.train['Date'].dt.dayofweek

        self.test['Date'] = pd.to_datetime(self.test['Date'])
        self.test['Year'] = self.test['Date'].dt.year
        self.test['Month'] = self.test['Date'].dt.month
        self.test['Day'] = self.test['Date'].dt.day
        self.test['DayOfWeek'] = self.test['Date'].dt.dayofweek

        self.assertIn('Year', self.train.columns, "Year feature is missing in train dataset")
        self.assertIn('Month', self.train.columns, "Month feature is missing in train dataset")
        self.assertIn('Day', self.train.columns, "Day feature is missing in train dataset")
        self.assertIn('DayOfWeek', self.train.columns, "DayOfWeek feature is missing in train dataset")

    def test_model_training(self):
        # Feature engineering
        self.train['Weekend'] = self.train['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        self.test['Weekend'] = self.test['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        holidays = pd.to_datetime(['2013-12-25', '2014-12-25', '2015-12-25'])

        self.train['DaysToHoliday'] = self.train['Date'].apply(lambda x: (holidays[holidays > x].min() - x).days if (holidays > x).any() else 0)
        self.train['DaysSinceHoliday'] = self.train['Date'].apply(lambda x: (x - holidays[holidays < x].max()).days if (holidays < x).any() else 0)

        self.test['DaysToHoliday'] = self.test['Date'].apply(lambda x: (holidays[holidays > x].min() - x).days if (holidays > x).any() else 0)
        self.test['DaysSinceHoliday'] = self.test['Date'].apply(lambda x: (x - holidays[holidays < x].max()).days if (holidays < x).any() else 0)

        self.train['BeginningOfMonth'] = self.train['Day'].apply(lambda x: 1 if x <= 10 else 0)
        self.train['MidMonth'] = self.train['Day'].apply(lambda x: 1 if 10 < x <= 20 else 0)
        self.train['EndOfMonth'] = self.train['Day'].apply(lambda x: 1 if x > 20 else 0)

        self.test['BeginningOfMonth'] = self.test['Day'].apply(lambda x: 1 if x <= 10 else 0)
        self.test['MidMonth'] = self.test['Day'].apply(lambda x: 1 if 10 < x <= 20 else 0)
        self.test['EndOfMonth'] = self.test['Day'].apply(lambda x: 1 if x > 20 else 0)

        features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2', 'Year', 'Month', 'Day', 'Weekend', 'DaysToHoliday', 'DaysSinceHoliday', 'BeginningOfMonth', 'MidMonth', 'EndOfMonth']
        target = 'Sales'

        X = self.train[features]
        y = self.train[target]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)

        self.assertTrue(mae < 2000, f"Mean Absolute Error is too high: {mae}")

    def test_lstm_model(self):
        # Check LSTM model exists
        try:
            model = load_model('lstm_model.h5')
        except Exception as e:
            self.fail(f"Failed to load LSTM model: {e}")

        # Check if the model can make a prediction
        daily_sales = self.train.groupby('Date')['Sales'].sum().reset_index()
        daily_sales['Sales_diff'] = daily_sales['Sales'].diff().dropna()
        window_size = 30
        series = daily_sales['Sales_diff'].dropna().values
        batch = series[-window_size:].reshape((1, window_size, 1))
        
        try:
            pred = model.predict(batch)
            self.assertIsInstance(pred, np.ndarray, "LSTM model prediction is not a numpy array")
            self.assertEqual(pred.shape, (1, 1), "LSTM model prediction shape is incorrect")
        except Exception as e:
            self.fail(f"Failed to make prediction with LSTM model: {e}")

if __name__ == '__main__':
    unittest.main()