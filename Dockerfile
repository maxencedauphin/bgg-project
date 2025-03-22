FROM python:3.10-slim

# COPY models models
COPY bgg-project bgg-project
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --upgrade pip
RUN pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org -e .

# Run container locally
CMD uvicorn bgg-project.api_file:app --reload --host 0.0.0.0

# Run container deployed -> GCP
# CMD uvicorn bgg-project.api_file:app --reload --host 0.0.0.0 --port $PORT

