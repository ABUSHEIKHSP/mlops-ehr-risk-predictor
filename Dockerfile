FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY models/ models/
COPY src/ src/

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.app.predict:app", "--host", "0.0.0.0", "--port", "8000"]
