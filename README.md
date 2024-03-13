Simple python-image with 2 scripts:
1. Read pdf-files in DATA_PATH into chroma vector-db
2. Query OpenAI-Api with results from db as context


Examlpe compose-file:
```yaml

version: '3'
services:
  app:
    build:
      context: .
    #volumes:
    #  - ./mount:/app
    command: tail -f /dev/null
    environment:
      - DATA_PATH=${DATA_PATH}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                device_ids: ['0']
                capabilities: [gpu]
```

create .env-file:
```shell
cp .env.example .env
```
then insert your Openai-api-key


Start container:
```shell
docker compose up -d
```

Start container:
```shell
docker compose up -d
```