FROM python:3.8.9
RUN mkdir -p /app && \
    mkdir -p /app/client && \
    mkdir -p /app/certs && \
    mkdir -p /app/client/certs && \
    chmod g+rwx /app/client/certs && \
    mkdir -p /app/client/package && \
    chmod g+rwx /app/client/package
COPY fedn /app/fedn
RUN pip install -e /app/fedn
WORKDIR /app
