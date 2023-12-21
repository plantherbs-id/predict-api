FROM python:3.11.0

WORKDIR /predict-api 

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# ENV PORT=8080

EXPOSE 5000

CMD ["python", "main.py"]

