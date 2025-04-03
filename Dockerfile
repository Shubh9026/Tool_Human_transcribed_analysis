# Use an official Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first (for caching layers)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# First, run video_processing.py, then start app.py
CMD ["sh", "-c", "python video_processing.py && python app.py"]
