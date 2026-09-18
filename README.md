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

### `GET /models`

Endpoint di discovery: il servizio carica qualsiasi modello su richiesta, quindi
non c'è un elenco fisso di modelli ammessi. Restituisce il default, i modelli
attualmente residenti e le famiglie che supportano `input_type`, così il client
sa quando `input_type` conta.

```json
{
  "default_model": "intfloat/multilingual-e5-small",
  "loaded_models": ["intfloat/multilingual-e5-small"],
  "model_cache": 2,
  "input_types": ["query", "passage"],
  "prefix_families": {
    "e5": { "query": "query: ", "passage": "passage: " },
    "nomic": { "query": "search_query: ", "passage": "search_document: " }
  }
}
```

### `POST /embed`

Genera gli embeddings. Body JSON:

- `text` (string) **oppure** `texts` (array di string) — obbligatorio
- `model` (string, opzionale) — override del modello di default
- `input_type` (`"query"` | `"passage"`, opzionale) — intento **semantico**:
  `query` = sto cercando, `passage` = sto indicizzando. Il server lo traduce nel
  prefisso corretto per il modello (es. e5 → `query:` / `passage:`); per i
  modelli che non usano prefissi viene ignorato. Se omesso, nessun prefisso.
- `normalize_embeddings` (bool, opzionale, default `true`)

Il client resta **agnostico rispetto al modello**: manda sempre la stessa forma
e lascia al server la conoscenza di come si parla a ciascun modello.

```bash
curl -s http://localhost:8000/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["ciao mondo", "hello world"], "input_type": "passage"}'
```

Risposta:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "input_type": "passage",
  "prefix_applied": true,
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
