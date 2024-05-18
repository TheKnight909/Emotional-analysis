# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file separately to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
# Install nltk and download the stopwords
RUN python -c "import nltk; nltk.download('stopwords')"

# Copy the rest of the application code
COPY . .

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Health check for Docker
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Command to run the Streamlit app on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
