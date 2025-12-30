# Template Structure - Self-Contained Projects

Each template is a **completely independent project** with its own requirements, config, and code.

## Independent Folders

Each concept/library has its own folder:

```
_Templates/
├── Template_Orbit_Python/          # Orbit - Bayesian forecasting
│   ├── main.py
│   ├── config.yaml
│   ├── requirements.txt            # orbit-ml, pandas, numpy, matplotlib, pyyaml
│   ├── plotting_utils.py
│   ├── README.md
│   ├── data/
│   └── outputs/
│
├── Template_Aeon_Python/            # Aeon - TS toolkit
│   ├── main.py
│   ├── config.yaml
│   ├── requirements.txt            # aeon-toolkit, pandas, numpy, matplotlib, pyyaml
│   ├── plotting_utils.py
│   ├── README.md
│   ├── data/
│   └── outputs/
│
├── Template_Kalman_Python/          # Kalman filters
│   ├── main.py
│   ├── config.yaml
│   ├── requirements.txt            # filterpy, pandas, numpy, matplotlib, pyyaml
│   ├── plotting_utils.py
│   ├── README.md
│   ├── data/
│   └── outputs/
│
└── Template_Merlion_Python/          # Merlion - forecasting & anomaly
    ├── main.py
    ├── config.yaml
    ├── requirements.txt            # salesforce-merlion, pandas, numpy, matplotlib, pyyaml
    ├── plotting_utils.py
    ├── README.md
    ├── data/
    └── outputs/
```

## Key Points

1. **Each folder is independent** - No shared requirements file
2. **Each has its own requirements.txt** - Only the packages needed for that library
3. **Self-contained** - Can copy any folder and use it standalone
4. **No dependencies between templates** - Each works independently

## Usage

To use any template:

```bash
# Copy the template you want
cp -r Template_Orbit_Python my_orbit_project

# Navigate to it
cd my_orbit_project

# Install ONLY its requirements
pip install -r requirements.txt

# Run it
python main.py
```

## Requirements Summary

| Template | Main Library | Other Dependencies |
|----------|-------------|-------------------|
| Orbit | orbit-ml | pandas, numpy, matplotlib, pyyaml |
| Aeon | aeon-toolkit | pandas, numpy, matplotlib, pyyaml |
| Kalman | filterpy | pandas, numpy, matplotlib, pyyaml |
| Merlion | salesforce-merlion | pandas, numpy, matplotlib, pyyaml |

Each template only installs what it needs - no mega requirements file!

