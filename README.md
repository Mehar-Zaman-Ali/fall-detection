# Fall Detection ML Model

Sensor-based fall detection system using the Smartphone Human Fall Dataset. Classifies smartphone sensor readings as **Fall** or **Non-Fall (ADL)** using Random Forest (primary) and 1D CNN (secondary).

## Dataset

The Smartphone Human Fall Dataset contains pre-extracted features from accelerometer and gyroscope sensors:

| Feature | Description |
|---|---|
| `acc_max` | Peak acceleration magnitude during the event |
| `acc_kurtosis` | Kurtosis of acceleration signal |
| `acc_skewness` | Skewness of acceleration signal |
| `gyro_max` | Maximum angular velocity (removed during feature selection) |
| `gyro_kurtosis` | Kurtosis of gyroscope signal |
| `gyro_skewness` | Skewness of gyroscope signal |
| `lin_max` | Maximum linear acceleration |
| `post_lin_max` | Maximum linear acceleration post-impact |
| `post_gyro_max` | Maximum angular velocity post-impact |

**Activity Labels**: SDL, FOL, FKL, BSC (falls) and JOG, JUM, WAL, STD, CSI, CSO, STN (daily activities).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Open and run `fall_detection.ipynb` in Jupyter Notebook or VS Code.

```bash
jupyter notebook fall_detection.ipynb
```

## Pipeline

1. **Data Loading** - download and explore the dataset
2. **EDA** - mutual information scores, correlation heatmap, swarm plots, skewness analysis
3. **Feature Engineering** - remove noisy features, normalize, train/test split
4. **Random Forest** - train, evaluate, feature importance
5. **1D CNN** - train, evaluate, learning curves
6. **Comparison** - side-by-side model performance
