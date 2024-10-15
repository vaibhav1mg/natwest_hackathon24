# Use an official Python runtime as a parent image
FROM python:3.11.10-slim

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project folder into the container, excluding files listed in .dockerignore
COPY . .

# Expose necessary ports
EXPOSE 8501
EXPOSE 8000

# Install supervisor to manage multiple processes
RUN apt-get update && apt-get install -y supervisor

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Command to run supervisord
CMD ["/usr/bin/supervisord"]
