# Breast Cancer Prediction App
## How to Run the code (if using EC2)

### Note: Do the port mapping to this port:- 8501
### Note: Due to the nature of the breast cancer data, it includes 30 ENCRYPTED features and we do not know their meaning. Therefore, it does not make sense to make the value of these features adjustable, like what we did in the iris app. Therefore, I provide the user with 2 examples, one for Negative case, and one for Positive case, and the user can choose one of them to see if the model works OK. (It though the dataset is about image analysis, but it's unfortunately not)

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