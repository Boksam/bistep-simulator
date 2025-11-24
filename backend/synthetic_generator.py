import numpy as np
import pandas as pd


class SyntheticDataGenerator:

    def generate(self,
                 pattern_type: str,
                 params: dict,
                 start_date: str,
                 end_date: str,
                 interval: str = "1h"):
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        # Normalize time to 0..10 for easier parameter tuning
        x = np.linspace(0, 10, len(dates))

        if pattern_type == "polynomial":
            # params: degree, coefficients (list or dict)
            # We expect coefficients as a list [a, b, c] for ax^2 + bx + c
            coeffs = params.get("coefficients", [0])
            model = np.poly1d(coeffs)
            y = model(x)

        elif pattern_type == "exponential":
            # y = a * e^(b*x) + c
            a = params.get("scale", 1.0)
            b = params.get("rate", 0.5)
            c = params.get("offset", 0.0)
            y = a * np.exp(b * x) + c

        elif pattern_type == "step":
            # Sigmoid function
            # y = min_val + (max_val - min_val) / (1 + exp(-speed * (x - step_position)))
            min_val = params.get("min_val", 0.0)
            max_val = params.get("max_val", 10.0)
            step_pos = params.get("step_position", 5.0)
            speed = params.get("transition_speed", 2.0)

            y = min_val + (max_val - min_val) / (1 + np.exp(-speed *
                                                            (x - step_pos)))

        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        # Add noise
        noise_std = params.get("noise_std", 0.0)
        if noise_std > 0:
            y += np.random.normal(0, noise_std, len(y))

        return pd.DataFrame({"timestamp": dates, "value": y})
