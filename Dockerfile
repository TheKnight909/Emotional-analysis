# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/streamlit/streamlit-example.git .

# Copy requirements.txt separately to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies first to leverage Docker cache
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy the .streamlit directory to the working directory
COPY .streamlit /app/.streamlit

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Health check for Docker
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Command to run the Streamlit app on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
