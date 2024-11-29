FROM python:3.10
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY source dest

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]