FROM python:3.11

# Evitar pyc y buffering de stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    pkg-config \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["sleep", "infinity"]
# CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
# jupyter notebook --ip=0.0.0.0 --port=8501 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
# streamlit run app/src/streamlit/eicu_patient_dashboard.py
