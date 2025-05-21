FROM python:3.12

WORKDIR /opt/ml_in_app

COPY requirements_prod.txt ./
RUN pip install --upgrade pip && pip install -r requirements_prod.txt

COPY . /opt/ml_in_app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
