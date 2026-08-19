# src/predict.py

from pathlib import Path
import joblib
import pandas as pd


# Path to the trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "mobile_price_model.pkl"

# These must match the features used during training
FEATURES = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]


def load_model():
    """Load the trained ML pipeline."""
    return joblib.load(MODEL_PATH)


def predict_price_range(features: dict) -> int:
    # Check that all required features are present
    missing = [feature for feature in FEATURES if feature not in features]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Keep the feature order identical to training
    input_data = pd.DataFrame(
        [[features[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    model = load_model()

    prediction = model.predict(input_data)

    return int(prediction[0])


if __name__ == "__main__":

    phone = {
        "battery_power": 1800,
        "blue": 1,
        "clock_speed": 2.2,
        "dual_sim": 1,
        "fc": 8,
        "four_g": 1,
        "int_memory": 64,
        "m_dep": 0.5,
        "mobile_wt": 150,
        "n_cores": 8,
        "pc": 12,
        "px_height": 1080,
        "px_width": 2400,
        "ram": 3200,
        "sc_h": 15,
        "sc_w": 7,
        "talk_time": 15,
        "three_g": 1,
        "touch_screen": 1,
        "wifi": 1,
    }

    result = predict_price_range(phone)

    print(f"Predicted price range: {result}")