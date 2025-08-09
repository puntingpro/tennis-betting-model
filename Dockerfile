# --- Build Stage ---
# Use a specific, stable version of the base image to ensure reproducibility
FROM python:3.11.9-slim-bullseye as builder

# Apply security updates to the base image
RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
# Use the same minimal base image for the final container
FROM python:3.11.9-slim-bullseye

WORKDIR /app

# Create a non-root user and switch to it for added security
RUN useradd --create-home appuser
USER appuser

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application code
COPY . .

# Install the application
RUN pip install .

# Define the command to run your application
CMD ["python", "main.py", "stream"]
