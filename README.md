# sentence-api

Piccolo servizio HTTP (Flask) che espone embeddings tramite
[sentence-transformers](https://www.sbert.net/). Usato dallo stack Laraplate:
il client invia in ogni richiesta il `service_model` del profilo attivo, quindi
il modello di default serve solo quando la richiesta non specifica `model`.

## Requisiti

- Python 3.10+
- Le dipendenze in [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configurazione (variabili d'ambiente)

| Variabile                | Default                              | Descrizione                                   |
| ------------------------ | ------------------------------------ | --------------------------------------------- |
| `EMBEDDING_MODEL`        | `intfloat/multilingual-e5-small`     | Modello caricato all'avvio / usato di default |
| `EMBEDDING_MODEL_CACHE`  | `2`                                  | Quanti modelli tenere residenti (LRU)         |

I modelli vengono scaricati automaticamente da Hugging Face al primo utilizzo e
non sono versionati nel repo.

## Avvio

```bash
python sentence-api.py
```

Il servizio ascolta su `0.0.0.0:8000`.

## API

### `GET /health`

Stato del servizio e modelli caricati.

```json
{
  "status": "healthy",
  "model": "intfloat/multilingual-e5-small",
  "default_model": "intfloat/multilingual-e5-small",
  "loaded_models": ["intfloat/multilingual-e5-small"],
  "model_cache": 2
}
```

### `POST /embed`

Genera gli embeddings. Body JSON:

- `text` (string) **oppure** `texts` (array di string) — obbligatorio
- `model` (string, opzionale) — override del modello di default
- `normalize_embeddings` (bool, opzionale, default `true`)

```bash
curl -s http://localhost:8000/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["ciao mondo", "hello world"]}'
```

Risposta:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "embeddings": [[...], [...]]
}
```

## Deploy (systemd)

Il file [`sentence-transformers.service`](sentence-transformers.service) è la
unit usata in produzione. Presuppone il virtualenv in `/opt/ai-env` e il codice
in `/opt/sentence-api`.

```bash
sudo cp sentence-transformers.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentence-transformers.service
```
