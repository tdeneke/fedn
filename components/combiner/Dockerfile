FROM python:3.8.9
RUN mkdir -p /app && \
    mkdir -p /app/client &&\
    mkdir -p /app/certs
COPY fedn /app/fedn
RUN pip install -e /app/fedn
WORKDIR /app
