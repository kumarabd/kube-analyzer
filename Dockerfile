FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create non-root user
RUN groupadd -g 1000 kube-analyzer && \
    useradd -u 1000 -g kube-analyzer -s /bin/bash -m kube-analyzer

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Trivy for vulnerability scanning
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - && \
    echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | tee -a /etc/apt/sources.list.d/trivy.list && \
    apt-get update && \
    apt-get install -y trivy && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Make main.py executable
RUN chmod +x /app/src/main.py

# Create reports directory with proper permissions
RUN mkdir -p /app/reports && chown -R kube-analyzer:kube-analyzer /app/reports

# Switch to non-root user
USER kube-analyzer

# Set Python path
ENV PYTHONPATH=/app

# Command to run
ENTRYPOINT ["python", "/app/src/main.py"]

# Default arguments
CMD ["--in-cluster", "--analysis-type", "all"]
