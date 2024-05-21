# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 6001

# Define environment variable
# ENV MODEL_NAME=gpt-3.5-turbo

# Run llm_backend.py when the container launches
CMD ["python", "llm_backend.py"]
