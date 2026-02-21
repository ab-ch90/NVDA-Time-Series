# NVDA-Time-Series

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py
python src/train_model.py
python src/predict.py
python src/plot.py