# Nail Disease Classifier

Simple prototype that classifies nail images (6 classes) using a pretrained ResNet18, shows Grad‑CAM explainability, and reports a simple severity heuristic.

## Quick setup

1. Create and activate a virtual environment (Windows PowerShell):
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your trained model file `nail_classifier_resnet18.pth` in the project root (or update paths in `app.py`/`nail.py`).

4. Run the web UI:
   ```
   python app.py
   ```
   Open http://127.0.0.1:5000

## Notes & security
- Do NOT run Flask with `debug=True` in production.
- Validate uploaded files and use `werkzeug.utils.secure_filename`; add `app.config['MAX_CONTENT_LENGTH']` to limit upload size.
- Model and Grad‑CAM hooks should be managed to avoid memory leaks; consider consolidating shared logic into a single module.
- For CUDA users, install a torch wheel appropriate for your CUDA version. See https://pytorch.org/get-started/locally/.

## Files of interest
- `app.py` — Flask app / web UI.
- `nail.py` — training / CLI utilities and local prediction helpers.
- `templates/` — `index.html`, `result.html`.