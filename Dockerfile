FROM python:3.10
COPY . /usr/lendsqr/
EXPOSE 5000
WORKDIR /usr/lendsqr/
COPY requirements.txt /usr/lendsqr/
RUN pip install --default-timeout=1000 -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "flaskapp:lensqr"]