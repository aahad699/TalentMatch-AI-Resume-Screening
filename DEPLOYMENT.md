# Deployment Guide

This project is ready to deploy as a Streamlit app.

## Recommended: Streamlit Community Cloud

1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from your GitHub repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Keep `requirements.txt` in the repository root. Streamlit Community Cloud
   uses dependency files such as `requirements.txt` to install app packages.

## Alternative: Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Select Streamlit as the SDK.
3. Upload or connect this repository.
4. Use `streamlit_app.py` as the app entrypoint.

Hugging Face Spaces is suitable for hosting ML portfolio demos and sharing a
public project link.

## Before Sharing Publicly

- Do not upload private resumes.
- Keep demo data synthetic.
- Mention that the included model is trained on synthetic labeled data.
- Replace synthetic training data with representative labeled pairs before
  claiming production performance.

## Local Run

```powershell
.\run_app.ps1
```

Open:

- App: http://127.0.0.1:8501
- API docs: http://127.0.0.1:8000/docs
