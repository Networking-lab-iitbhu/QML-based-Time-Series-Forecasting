# Use a base image with the necessary dependencies
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy your application files to the container
COPY . /app

# Install any dependencies
RUN pip install -r requirements.txt

# Set a default command
CMD ["python", "main.py"]
