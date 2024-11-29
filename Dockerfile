FROM python:3.10-slim

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies untuk OpenCV dan easyOCR (libGL)
RUN apt-get update && \
  apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 && \
  rm -rf /var/lib/apt/lists/*

#Install phyton dependencies
COPY requirements.txt /app/

RUN pip install -r requirements.txt
RUN pip install "fastapi[standard]"

COPY . .

# Expose port 8080
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]