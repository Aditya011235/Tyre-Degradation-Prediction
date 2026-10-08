import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset shaped like the real one, for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 200
    events = rng.choice(["Monaco GP", "Silverstone GP", "Monza GP"], size=n)
    tire_wear = rng.uniform(0, 1, size=n)
    humidity = rng.uniform(20, 90, size=n)
    ambient_temperature = rng.uniform(10, 40, size=n)
    noise = rng.normal(0, 0.5, size=n)
    target = 10 * tire_wear + 0.05 * humidity + 0.1 * ambient_temperature + noise

    return pd.DataFrame(
        {
            "Tire_wear": tire_wear,
            "Humidity": humidity,
            "Ambient_Temperature": ambient_temperature,
            "Event": events,
            "Tire_Degradation": target,
        }
    )
