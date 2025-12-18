## OC-P11

## POC Chatbot RAG

**Table of contents**

- [Getting started](#getting-started)
	- [Download Docker](#download-docker)
	- [Download and vectorize datas](#download-and-vectorize-datas)
	- [Launch the chatbot service](#launch-the-chatbot-service)

## Getting started

### Download Docker
Download the Docker Desktop mathing your OS on https://www.docker.com/products/docker-desktop/

### Download and vectorize datas
To download data from OpenDataSoft and vectorize them with Mistral API using, use the command line below.
If you already did this step, you can skip it unless you want to refresh the dataset.

Default :
```bash
    docker compose --profile vb up -d
```

### Launch the chatbot service
Use the command line below.

Default :
```bash
    docker compose --profile chatbot up -d
```
