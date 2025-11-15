FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
ENV PORT=8000
EXPOSE 8000
CMD gunicorn -w 2 -b 0.0.0.0:$PORT app:app
