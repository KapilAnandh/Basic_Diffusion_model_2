# Dockerfile for SDXL Gradio app
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port for Gradio
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
