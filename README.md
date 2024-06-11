# Breast Cancer Prediction App
## How to Run the code (if using EC2)
```bash
sudo apt update
```

```bash
sudo apt-get update
```

```bash
sudo apt upgrade -y
```

```bash
sudo apt install git curl unzip tar make sudo vim wget -y
```

```bash
git clone https://github.com/MSIA/423-2024-hw3-iyq5197.git
```

```bash
sudo apt install python3-pip
```

```bash
python3 -m venv myenv
source myenv/bin/activate
```

```bash
cd 423-2024-hw3-iyq5197
pip3 install -r requirements.txt
```

you probably don't need to load since we already have these models loaded
```bash
python load.py
````

Temporary running
```bash
python3 -m streamlit run app.py
```
Permanent running
```bash
nohup python3 -m streamlit run app.py
```

Note: Streamlit runs on this port: 8501

# Docker ECS

## Build the Docker image

```bash
docker build -t cancer-app -f dockerfiles/Dockerfile .
```

## Run the entire model pipeline

```bash
docker run -p 85:01 -v $(pwd)/artifacts:/app/artifacts cancer-app
```
