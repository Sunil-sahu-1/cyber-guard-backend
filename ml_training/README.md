# Cyber Guard ML Training Pack

This directory contains reproducible dataset download and model-training entry points for Cyber Guard.

## Models

| Module | Dataset | Model |
|---|---|---|
| Voice deepfake | ASVspoof 2019 LA + ASVspoof 2021 DF | AASIST-L / AASIST |
| Voice replay | ASVspoof 2021 PA | AASIST-style anti-spoofing pipeline |
| Image deepfake | FaceForensics++ | EfficientNet-B0 transfer learning |
| Video deepfake | DFDC + FaceForensics++ | EfficientNet-B0 frame model + video aggregation |
| Network anomaly | CIC-IDS2017 | XGBoost |
| Malware | EMBER 2018 | LightGBM |
| Phishing URL | PhishTank + benign URL corpus | LightGBM |
| Phishing email | SpamAssassin + phishing email corpus | TF-IDF + Logistic Regression baseline |
| Login/behavior anomaly | Cyber Guard application logs | Isolation Forest |

These are practical training choices for the current Cyber Guard codebase; they are not a claim that one architecture is universally state-of-the-art.

## Important

The datasets themselves are NOT committed to GitHub. Several are multi-GB, and FaceForensics++ requires acceptance of its terms. The download script creates the local dataset folders and downloads only sources that can be downloaded without an interactive access step.

After downloading, train with:

    python -m ml_training.train_all

Or train one module:

    python -m ml_training.train_tabular --network
    python -m ml_training.train_tabular --malware
    python -m ml_training.train_tabular --phishing-url

For deepfake and voice training, use the dedicated commands described by the script:

    python -m ml_training.train_media --image
    python -m ml_training.train_media --video
    python -m ml_training.train_voice --track DF

Outputs are stored under:

    trained_models/
    training_results/

Both folders are ignored by Git.
