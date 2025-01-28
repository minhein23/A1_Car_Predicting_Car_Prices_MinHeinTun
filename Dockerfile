# Use Python as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Expose the Flask app's port
EXPOSE 5000

# Command to run the Flask app
CMD ["python3", "app.py"]