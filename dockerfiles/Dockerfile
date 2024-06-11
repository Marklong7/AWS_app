FROM --platform=linux/arm64 python:3.11-slim

# Set the working directory
WORKDIR /cancer_app

RUN /usr/local/bin/python -m pip install --upgrade pip

# requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# code the code
COPY . .

EXPOSE 60

ENV PYTHONUNBUFFERED=1

# Set the command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.fileWatcherType=none", "--server.port=8501"]