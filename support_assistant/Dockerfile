FROM python:3.11-slim

WORKDIR /app
COPY . /app/support_assistant
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

ENV MOCK_LLM=1
EXPOSE 7860

CMD ["uvicorn","support_assistant.main:app","--host","0.0.0.0","--port","7860"]
