import pickle
import os
from dotenv import load_dotenv
import unittest
from unittest.mock import patch, MagicMock
from app import *
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Define the artifacts directory
artifacts_dir_name = "artifacts"
artifacts_dir = Path(artifacts_dir_name)

# Load environment variables from .env
load_dotenv()

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = {
            "model_lasso": MagicMock(),
            "model_logistic": MagicMock()
        }

    def test_load_models_success(self):
        with patch("app.Path.glob") as mock_glob:
            mock_glob.return_value = [Path("artifacts/model_lasso.pkl"), Path("artifacts/model_logistic.pkl")]
            models = load_models()
            self.assertEqual(len(models), 2)

    def test_load_models_exception(self):
        with patch("app.Path.glob") as mock_glob:
            mock_glob.return_value = [Path("artifacts/model_lasso.pkl"), Path("artifacts/model_logistic.pkl")]
            with patch("builtins.open") as mock_open:
                mock_open.side_effect = Exception("File not found")
                models = load_models()
                self.assertEqual(len(models), 0)

    def test_model_selection(self):
        with patch("streamlit.sidebar.selectbox") as mock_selectbox:
            mock_selectbox.return_value = "model_lasso"
            selected_model = model_selection(self.models)
            self.assertEqual(selected_model, "model_lasso")

    def test_load_selected_model_found(self):
        selected_model = "model_lasso"
        model = load_selected_model(selected_model, self.models)
        self.assertIsNotNone(model)

    def test_load_selected_model_not_found(self):
        selected_model = "model_nonexistent"
        with patch("app.st.error") as mock_error:
            model = load_selected_model(selected_model, self.models)
            self.assertIsNone(model)
            mock_error.assert_called_once()

def main():
    unittest.main()

if __name__ == "__main__":
    main()
