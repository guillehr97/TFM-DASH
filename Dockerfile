FROM python:3.8

# set the working directory in the container
WORKDIR /code
# copy the dependencies file to the working directory
COPY requirements.txt .
COPY iex_dow.csv .
COPY modelo_2d_lineas1.h5 .
COPY modelo1D_3.h5 .
COPY predicciones.csv .

# install dependencies
RUN pip install -r requirements.txt


CMD [ "python", "./DASH BUENO.py" ]
