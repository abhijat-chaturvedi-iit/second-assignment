#FROM ubuntu:latest
FROM python:3.8.1
COPY ./* /exp/
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
# CMD ["python", "./plot_digits_classification.py"]
CMD ["python","./q3.py", "--clf_name","%s","--random_state","%s"]
# EXPOSE 5000
RUN echo "Done"