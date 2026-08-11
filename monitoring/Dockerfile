FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY monitoring/requirements.txt ./monitoring/requirements.txt
RUN pip install --no-cache-dir -r monitoring/requirements.txt

COPY common ./common
COPY monitoring ./monitoring

EXPOSE 8501

CMD ["streamlit", "run", "monitoring/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
