# Use PyTorch CPU-only image
FROM pytorch/pytorch:2.6.0

# Set working directory inside the container
WORKDIR /pipeline

# Copy the entire project into the container (assumes Dockerfile is at project root)
COPY .. .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app located in the pipeline folder
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
