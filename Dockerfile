FROM python:3.10-slim

# Sets the working directory
WORKDIR /app

# Copies the current directory contents into the container
COPY . /app

# Installs Python packages from requirements.txt
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python3", "application.py"]