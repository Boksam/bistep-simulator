# Data Generation Flow

This diagram illustrates the process of generating synthetic sensor data in the BISTEP simulator.

```mermaid
graph TD
    A[Start Simulation] --> B{Use Historical Data?};
    B -- Yes --> C[Load Historical Data];
    C --> D[Preprocess Data];
    D --> E[STL Decomposition];
    E --> F[Extract Trend Model];
    E --> G[Extract Seasonality];
    E --> H[Extract Noise Model];
    
    B -- No --> I[User Defined Parameters];
    I --> F;
    I --> G;
    I --> H;
    
    F --> J[Generate Trend Component];
    G --> K[Generate Seasonality Component];
    H --> L[Generate Noise Component];
    
    J --> M[Combine Components];
    K --> M;
    L --> M;
    
    M --> N[Final Synthetic Data];
    N --> O[End];

    subgraph "Decomposition & Modeling"
    E
    F
    G
    H
    end

    subgraph "Synthesis"
    J
    K
    L
    M
    end
```

## Component Details

1.  **Trend ($T_t$)**: Modeled using Polynomial Regression (Degree 1 or 2).
2.  **Seasonality ($S_t$)**:
    *   **Yearly**: Average pattern over a year.
    *   **Daily**: Average pattern over a day.
3.  **Noise ($R_t$)**: Modeled using AutoRegressive (AR) models or Gaussian Noise.

## Formula

$$ Y_{new}(t) = T_{poly}(t) + S_{yearly}(t) + S_{daily}(t) + \text{Noise}_{AR}(t) $$
