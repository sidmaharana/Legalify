# Use a slim official Python image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies efficiently by copying only requirements.txt first
# This leverages Docker's layer caching: if requirements.txt doesn't change,
# this layer won't be rebuilt, speeding up subsequent builds.
COPY backend/requirements.txt .
RUN apt-get update && apt-get install -y libomp5
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000, which is the port our Flask application runs on
EXPOSE 5000

# Define the command to run the application when the container starts
# This uses the exec form of CMD, which is recommended.
CMD ["python", "backend/app.py"]
