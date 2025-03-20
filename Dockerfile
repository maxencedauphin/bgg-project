FROM python:3.10-slim

COPY bgg_project bgg_project
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY API_package_folder API_package_folder

RUN pip install --upgrade pip
RUN pip install -e .

# Run container locally
#CMD uvicorn API_package_folder.api_file:app --reload --host 0.0.0.0

# Run container deployed -> GCP
CMD uvicorn API_package_folder.api_file:app --reload --host 0.0.0.0 --port $PORT
