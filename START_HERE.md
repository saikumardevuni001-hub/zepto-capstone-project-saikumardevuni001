# Start Here — Fastest Path

1. Open this folder in VS Code.
2. Open Terminal > New Terminal.
3. Create/activate a virtual environment:
   `python -m venv .venv`
   then Windows: `.venv\Scripts\activate`
4. Install:
   `pip install -r requirements.txt`
5. Run Module 1:
   `python data_pipeline/pipeline.py`
6. Run Module 2 on a machine with internet access:
   `python analytics/pipeline.py`
   This first creates the required `analytics/titanic.csv`; rerun after it finishes if you want to verify the offline path.
7. Run Module 3:
   `python -m support_assistant.ingest`
   then `uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860`
8. Before submission, commit the generated `analytics/titanic.csv`, generated outputs, and the database/artifacts that you want included.
9. Create the required Git branch/commit/merge history as shown in the root README.
10. Push the single `zepto_capstone_project` repository to GitHub as Public and submit that one URL.
