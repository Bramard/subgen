###
### Stage 0 - Base: Install base packages needed for builder and runtime
###
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

WORKDIR /subgen

# Install required build & runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ffmpeg \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Handle timezone from $TZ docker variable
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/timezone && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime

###
### Stage 1a - Builder: Install packages needed for builder only and build dependencies
###
FROM base AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Install only required build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

###
### Stage 1b - Builder: Load app source code
###
FROM builder AS builder-app

# Copy application source code (explicitely, some files may be missing compared to "COPY . ." original code)
COPY icon.png \
     language_code.py \
     launcher.py \
     subgen.py \
     subgen.xml \
     .

###
### Stage 2a - Runtime: Create a minimal runtime image
###
FROM base AS runtime

# Copy necessary files from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Install only required runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

###
### Stage 2b - Runtime: Load app source code and run app
###
FROM runtime AS runtime-app
# Copy application source code from builder-app
COPY --from=builder-app /subgen/launcher.py .
COPY --from=builder-app /subgen/subgen.py .
COPY --from=builder-app /subgen/language_code.py .

ENV PYTHONUNBUFFERED=1

# Set command to run the application
CMD ["python3", "launcher.py"]
