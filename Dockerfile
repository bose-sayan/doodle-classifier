FROM python:3.10

WORKDIR /

COPY . .

# install dependencies
RUN pip install -r requirements.txt

EXPOSE 5000
ENV FLASK_APP=server.py
# Command to run the Flask app
CMD ["flask", "run", "--host", "0.0.0.0"]
