FROM python
WORKDIR /cancer_app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the code
COPY . .

RUN tests

CMD ["test"]