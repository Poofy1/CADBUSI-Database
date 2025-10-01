ARG A="us-central1-docker.pkg.dev"
ARG B="aif-usr-p-walker-21e6"
ARG C="shared-aif-artifact-registry-docker-21e6"
ARG D="python:slim"
FROM "$D"
LABEL name=app.
LABEL version=1.0.0
ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    python3-venv \
    libssl-dev && \
    apt-get clean && \
    groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin -c "App User" appuser
ADD CertEmulationCA.crt /usr/local/share/ca-certificates/CertEmulationCA.crt
RUN update-ca-certificates
RUN pip config set global.cert /usr/local/share/ca-certificates/CertEmulationCA.crt
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /www
COPY --chown=appuser:appuser . /www
RUN chmod -R 755 /www
RUN pip install --no-cache-dir -r /www/requirements.txt
USER appuser
EXPOSE 5000
CMD ["gunicorn", \
    "--bind", ":5000", \
    "--workers", "1", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--threads", "2", \
    "app.main:app", \
    "--timeout", "0", \
    "--preload"]