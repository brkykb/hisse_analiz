# Use an official and lightweight Python base image
FROM python:3.11-slim

# Set environment variables to optimize Python in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Istanbul

# Install tzdata for accurate Turkish time tracking
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the keep-alive Flask server port
EXPOSE 8080

# Run the Telegram bot
CMD ["python", "hisse_bot.py"]
