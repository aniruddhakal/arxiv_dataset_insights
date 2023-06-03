FROM python:3.9
LABEL authors="ani"

ARG APP_NAME

RUN mkdir /app
COPY ./${APP_NAME} /app/${APP_NAME}

WORKDIR /app/$APP_NAME
RUN python -m pip install -r requirements.txt