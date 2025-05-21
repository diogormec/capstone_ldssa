# Usa imagem base do Python
FROM python:3.12

# Define o diretório de trabalho dentro do container
WORKDIR /opt/ml_in_app

# Copia os ficheiros para dentro do container
ADD . /opt/ml_in_app

# Instala as dependências
RUN pip install --upgrade pip
RUN pip install -r requirements_prod.txt

# Expõe a porta 5000 (ou a que usares)
EXPOSE 5000

# Define o comando de arranque
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
