 # use python
FROM python:3.9

# èsetup work directory
# Temporarily set the work directory to install dependencies
WORKDIR /usr/temp

# Copy requirements.txt to temp directory
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Now set the work directory to the mounted directory
WORKDIR /usr/algorithm


# expose code
EXPOSE 5000

# setup
CMD ["flask", "run", "--host=0.0.0.0"]
