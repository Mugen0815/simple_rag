Simple python-image with 2 scripts:
1. Read pdf-files in DATA_PATH into chroma vector-db
2. Query OpenAI-Api with results from db as context


Example compose-file:
```yaml
version: '3'

services:
  app:
    build:
      context: .
    command: tail -f /dev/null
    environment:
      - DATA_PATH=${DATA_PATH}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
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

Read files into db:
```shell
docker exec -it simple_rag_app_1 sh -c "python create_db.py"
```

Query db and openai-api:
```shell
docker exec -it simple_rag_app_1 sh -c "python query.py \"What is zend php?\""
```



