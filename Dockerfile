# Use an official Python runtime as a parent image
FROM alpine:3.5

RUN apk add --update py3-pip

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

# Copy the current directory contents into the container at /app
COPY main.py /usr/src/app/
COPY templates/form.html /usr/src/app/templates/

# Make port 80 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "/usr/src/app/main.py"]