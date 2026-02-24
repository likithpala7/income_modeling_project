# Census Income Classification and Segmentation

## Project Overview

This project trains an income classification model and a customer segmentation model on the US Census Bureau dataset. The classification model predicts whether a person earns above or below $50,000 per year using demographic and employment variables. The segmentation model groups individuals into distinct customer personas using K-Means clustering, providing actionable insights for targeted marketing campaigns.

---

## Repository Structure

```
income_modeling_project/
├── data/
│   ├── census-bureau.data
│   └── census-bureau.columns
├── src/
│   ├── classification.py
│   └── segmentation.py
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset is **not included** in this repository and must be obtained separately. You will need two files:

- `census-bureau.data` — the raw dataset
- `census-bureau.columns` — the column names file

Once obtained, place both files inside the `data/` folder at the root of the project. The `data/` directory must exist before running either script. If it does not exist, create it manually:

```bash
mkdir data
```

Then move your data files into it:

```bash
mv census-bureau.data data/
mv census-bureau.columns data/
```

---

## Environment Setup

> **Requirement:** Python 3.12.4 must be installed on your system before proceeding. You can verify your Python version with `python3.12 --version`.

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Navigate into the project directory

```bash
cd income_modeling_project
```

### 3. Create a virtual environment using Python 3.12.4

```bash
python3.12 -m venv venv
```

### 4. Activate the virtual environment

**Mac / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

Once activated, your terminal prompt will show `(venv)` at the beginning.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Classification Model

From the project root directory, run:

```bash
python src/classification.py
```

**Expected output:**
The script will print progress updates as it loads and preprocesses the data, trains the XGBoost model, and evaluates it on the held-out test set. Final output includes a full classification report with precision, recall, and F1 scores for both classes, as well as the weighted ROC-AUC score.

---

## Running the Segmentation Model

From the project root directory, run:

```bash
python src/segmentation.py
```

**Expected output:**
The script will print progress updates during preprocessing, PCA dimensionality reduction, and K-Means clustering. Silhouette scores for K=2 through K=9 will be printed to the terminal, followed by the cluster profile table showing mean statistics per segment. Two matplotlib plots will be displayed inline: a PCA scatter plot coloured by cluster assignment. Close each plot window to allow the script to continue.

---

## Requirements

The `requirements.txt` file pins all dependencies to the exact versions used during development with Python 3.12.4. Install them using the command in the Environment Setup section above.

```
asttokens==3.0.1
colorama==0.4.6
comm==0.2.3
contourpy==1.3.3
cycler==0.12.1
debugpy==1.8.20
decorator==5.2.1
executing==2.2.1
filelock==3.20.0
fonttools==4.61.1
fsspec==2025.12.0
imbalanced-learn==0.14.1
ipykernel==7.2.0
ipython==9.10.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.3
jupyter_client==8.8.0
jupyter_core==5.9.1
kiwisolver==1.4.9
MarkupSafe==3.0.2
matplotlib==3.10.8
matplotlib-inline==0.2.1
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.6.1
numpy==2.4.2
packaging==26.0
pandas==3.0.0
parso==0.8.6
pillow==12.1.1
platformdirs==4.9.2
prompt_toolkit==3.0.52
psutil==7.2.2
pure_eval==0.2.3
Pygments==2.19.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pyzmq==27.1.0
scikit-learn==1.8.0
scipy==1.17.0
seaborn==0.13.2
setuptools==70.2.0
six==1.17.0
sklearn-compat==0.1.5
stack-data==0.6.3
sympy==1.14.0
threadpoolctl==3.6.0
torch==2.10.0+cu126
torchvision==0.25.0+cu126
tornado==6.5.4
traitlets==5.14.3
typing_extensions==4.15.0
tzdata==2025.3
wcwidth==0.6.0
xgboost==3.2.0
```

---

## Troubleshooting

**Python 3.12 is not found (`python3.12: command not found`)**

Python 3.12.4 is not installed or not on your PATH. Download it from [https://www.python.org/downloads/release/python-3124/](https://www.python.org/downloads/release/python-3124/) and ensure it is added to your system PATH during installation. On Mac, you can also install it via Homebrew:
```bash
brew install python@3.12
```

**Data files are not found (`FileNotFoundError: census-bureau.data`)**

The script expects the data files at `../data/census-bureau.data` and `../data/census-bureau.columns` relative to the `src/` directory. Verify that both files exist in the `data/` folder at the project root and that the folder is named exactly `data` with no extra characters or spaces.

**Matplotlib plots do not display (headless / server environments)**

If running in a headless environment where a display is not available, matplotlib may throw a backend error. To fix this, add the following two lines to the top of `segmentation.py`, before any other imports:
```python
import matplotlib
matplotlib.use('Agg')
```
This switches matplotlib to a non-interactive backend that saves figures to files instead of displaying them. Change the `plt.show()` calls to `plt.savefig('output.png')` to save the plots to disk.