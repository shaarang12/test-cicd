FROM python:3.9-slim

WORKDIR /test-cicd

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GEMINI_API_KEY=${GEMINI_API_KEY}

EXPOSE 8080

CMD ["python", "app.py"]