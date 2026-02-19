FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project

COPY . .

RUN pip install --upgrade pip
RUN pip install pytest sphinx sphinx-autobuild

RUN pip install -e .

CMD ["pytest", "-v", "cvdp/tests/"]