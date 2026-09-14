# Voice Authentication Backend (Express + PostgreSQL + pgvector)

Node/Express API for user auth and voice enroll/verify file handling.

> **Note:** This service stores users and audio **file paths**. Speaker embeddings (192-dim ECAPA-TDNN vectors) are stored in PostgreSQL using the **pgvector** extension. Tables are created automatically on startup — you only need an empty database.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start API with nodemon |
| `npm start` | Start API once |
| `npm test` | Run unit tests (Jest) |
| `npm run migrate` | Apply SQL migrations in `migrations/` (creates schema + pgvector extension) |
| `npm run migrate:from-mysql` | One-off: copy data from an existing MariaDB/MySQL database into PostgreSQL |

## Unit tests

```bash
cd app/backend
npm test
```

Tests live under `tests/` and mock the PostgreSQL pool / ML client / filesystem (no live DB required to run `npm test`).

## Prerequisites

- **Node.js** 18+ (npm included)
- **PostgreSQL 13+** with the [pgvector](https://github.com/pgvector/pgvector) extension available

## 1) Start PostgreSQL + pgvector

### Option A — Docker (recommended)

A `docker-compose.yml` at the repo root starts a single Postgres service using the official `pgvector/pgvector` image (pgvector is prebuilt into the image, no manual extension install needed):

```bash
cd "speaker-verification-system"
docker compose up -d
```

This creates a `voice_authentication` database (or whatever `DATABASE_NAME` is set to) reachable on `localhost:5432`.

### Option B — Existing local PostgreSQL install

Install the `pgvector` extension for your PostgreSQL version (see the [pgvector install docs](https://github.com/pgvector/pgvector#installation)), then create a database:

```sql
CREATE DATABASE voice_authentication;
```

The app enables the extension itself (`CREATE EXTENSION IF NOT EXISTS vector`) on startup/migration — it does not need to be created manually as long as the extension files are installed on the server.

## 2) Configure `.env`

In `app/backend/`, create a file named `.env` (same folder as `package.json`), based on `.env.example`:

```env
PORT=5000
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=voice_auth
DATABASE_PASSWORD=your_postgres_password
DATABASE_NAME=voice_authentication
JWT_SECRET=replace_with_a_secure_secret
ML_SERVER_URL=http://localhost:8000
REQUIRED_ENROLLMENT_SAMPLES=3
```

Notes:
- Never commit `.env` (it is gitignored)
- Credentials are read only from environment variables — nothing is hardcoded

## 3) Install, migrate, and run

```bash
cd app/backend
npm install
npm run migrate   # creates the pgvector extension, tables, indexes
npm run dev
```

`npm run dev` / `npm start` also call the same schema setup on boot (`initSchema`, idempotent `CREATE ... IF NOT EXISTS`), so `npm run migrate` is optional but recommended as the explicit, reviewable migration step.

On success you should see:

```text
Server running on http://localhost:5000
```

Check health:

```text
GET http://localhost:5000/api/health
GET http://localhost:5000/api/health/db
```

### What gets created

| Table | Purpose |
|-------|---------|
| `users` | username + hashed password + role |
| `voice_samples` | enrollment/verification audio **paths** |
| `speaker_embeddings` | speaker embeddings as pgvector `vector(192)`; one `kind = 'template'` row per user holds the averaged enrollment embedding used for verification/identification (schema supports multiple embeddings per user) |
| `verification_logs` | verify **score / threshold / decision** history |
| `collection_samples` | research audio collected by admins (replay/live) |

## Migrating existing MariaDB/MySQL data

If you have an existing MariaDB/MySQL database from before this migration:

1. Make sure the old MySQL server is still reachable and PostgreSQL has the new schema (`npm run migrate`).
2. Set the source connection env vars (in `.env` or the shell) — these are separate from `DATABASE_*` so both databases can be reached at once:
   ```env
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_DATABASE=voice_authentication
   ```
3. Run:
   ```bash
   npm run migrate:from-mysql
   ```
   This copies `users`, `voice_samples`, `enrollment_templates` (converted from JSON to `vector(192)`), `verification_logs`, and `collection_samples`, preserving IDs and resyncing PostgreSQL sequences. It's idempotent (`ON CONFLICT DO NOTHING` on primary keys).

The MariaDB implementation is not deleted by this change — it simply isn't used once `DATABASE_*` env vars point the app at PostgreSQL.

### ML server (embeddings)

Speaker embeddings come from the Python service:

```powershell
cd D:\speaker-verification-system\app\server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Add to `.env`:

```env
ML_SERVER_URL=http://localhost:8000
REQUIRED_ENROLLMENT_SAMPLES=3
DEFAULT_VERIFY_THRESHOLD=0.25
IDENTIFY_THRESHOLD=0.25
IDENTIFY_MARGIN=0.05
REPLAY_DETECTION=true
```

After **3** enrollment uploads, Express calls `/enroll/template` and stores the average embedding as a `speaker_embeddings` row (`kind = 'template'`, pgvector `vector(192)`). Verify runs **replay detect** first (`/replay/detect`), then speaker match via `/verify` (unless `REPLAY_DETECTION=false`).

Voice login identify (`POST /api/voice/identify`) runs a pgvector cosine-distance nearest-neighbor query (`ORDER BY embedding <=> $1`, HNSW index) across every enrolled template instead of comparing in Node, then applies the same open-set gate: score must be ≥ `IDENTIFY_THRESHOLD`, and (when ≥2 templates) best−second ≥ `IDENTIFY_MARGIN`. Otherwise it returns 401 without a userid guess.

### Cosine distance vs. cosine similarity

pgvector's `<=>` operator (with `vector_cosine_ops`) returns **cosine distance**, defined as `1 - cosine_similarity`. The app's threshold (`DEFAULT_VERIFY_THRESHOLD`, `IDENTIFY_THRESHOLD`) is expressed in terms of cosine **similarity** (as it always was), so every SQL query that ranks embeddings selects `1 - (embedding <=> $1) AS score` to convert back to similarity before the existing `score >= threshold` comparison runs — the threshold value itself is unchanged.

## Common errors

| Error | Fix |
|-------|-----|
| `'nodemon' is not recognized` | Run `npm install` in `app/backend` first |
| `password authentication failed for user` | Check `DATABASE_USER` / `DATABASE_PASSWORD` in `.env` |
| `ECONNREFUSED` / cannot connect | PostgreSQL is not running (`docker compose up -d` or start your local service) |
| `database "voice_authentication" does not exist` | Create the database, or let `docker compose up -d` create it via `POSTGRES_DB` |
| `extension "vector" is not available` | Use the `pgvector/pgvector` Docker image, or install pgvector on your PostgreSQL server |
| ML enroll/verify errors / fetch failed | Start Python server on port **8000** (`app/server`) |
| `No enrollment template found` | Upload **3** enrollment samples first |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server with nodemon |
| `npm start` | Production-style `node server.js` |

## API Endpoints

Full request/response docs: see **[API.md](./API.md)**.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/auth/register` | No | Register user |
| `POST` | `/api/auth/login` | No | Login → JWT |
| `GET` | `/api/users/profile` | Bearer | Current user profile |
| `POST` | `/api/voice/enroll` | Bearer + `audio` file | Save enrollment WAV |
| `POST` | `/api/voice/verify` | Bearer + `audio` file | Verify + save score log |
| `GET` | `/api/voice/verification-logs` | Bearer | Verification score history |
| `GET` | `/api/voice/history` | Bearer | Sample history |
| `GET` | `/api/health` | No | Health check |
| `GET` | `/api/health/db` | No | Database health check |

CORS is configured for the Vite frontend at `http://localhost:5173`.

## Upload storage

Audio files are stored on **disk only** (not as BLOBs in PostgreSQL):

```text
uploads/enrollments/user_<id>/enroll_u<id>_s<n>_YYYYMMDD_HHMMSSmmm_<hex>.wav
uploads/verifications/user_<id>/verify_YYYYMMDD_HHMMSS.wav
```

PostgreSQL stores:
- relative `file_path` in `voice_samples`
- averaged ECAPA embedding (pgvector `vector(192)`) in `speaker_embeddings` (after 3 enroll uploads)
- verify `score` / `threshold` / `decision` in `verification_logs`

## Project layout

```text
backend/
├── server.js
├── config/
│   ├── db.js             # PostgreSQL pool + pgvector setup + auto schema
│   ├── migrate.js        # SQL migration runner (migrations/*.sql)
│   └── seed.js
├── migrations/            # SQL migration files (0001_init.sql, ...)
├── scripts/
│   └── migrateFromMySQL.js  # one-off MariaDB/MySQL -> PostgreSQL data copy
├── controllers/
├── models/
├── routes/
├── middleware/
├── uploads/
├── .env                  # local only — create this
├── package.json
└── README.md
```
