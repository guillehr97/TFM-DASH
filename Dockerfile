FROM python:3.8

# set the working directory in the container
WORKDIR /code

ENV PORT 8080
ENV HOST 0.0.0.0

# copy the dependencies file to the working directory
COPY requirements.txt .
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

COPY iex_dow.csv .
COPY modelo_2d_lineas1.h5 .
COPY modelo1D_3.h5 .
COPY dash_bucle.py .

# install dependencies
RUN pip install -r requirements.txt

CMD [ "python", "./dash_bucle.py" ]
