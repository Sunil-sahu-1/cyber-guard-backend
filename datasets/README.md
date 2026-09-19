# Cyber Guard datasets

Dataset files are intentionally excluded from Git.

Expected directories:

- voice/asvspoof2019_la
- voice/asvspoof2021_df
- voice/asvspoof2021_pa
- deepfake/faceforensics
- deepfake/dfdc
- deepfake/image_dataset
- deepfake/video_dataset
- network/cic_ids2017
- malware/ember
- phishing/urls
- phishing/emails
- behavior

Run:

    python -m ml_training.download_datasets

Then follow the printed official download instructions for datasets that
require terms acceptance or registration.
