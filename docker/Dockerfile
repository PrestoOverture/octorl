FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534
RUN pip install --no-cache-dir pytest==9.1.1 hypothesis==6.167.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
WORKDIR /workspace
CMD ["sleep", "infinity"]
