cd ~/projects/EmotionalCompany
source .venv/bin/activate

python -m pip install 'torch==2.10.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt

python -m pip check
python scripts/smoke_inference.py