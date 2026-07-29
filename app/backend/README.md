# Voice Authentication Backend (Express + MySQL)

Node/Express API for user auth and voice enroll/verify file handling.

> **Note:** This service stores users and audio **file paths**. Speaker models (ECAPA / replay) are planned as a separate **Python** service later. Tables are created automatically on startup — you only need to create the empty database.

## Prerequisites

- **Node.js** 18+ (npm included)
- **MySQL** or **MariaDB** running locally  
  Easy option on Windows: [XAMPP](https://www.apachefriends.org/) → start **MySQL**
- Optional GUI: [DBeaver](https://dbeaver.io/) (client only — does not replace MySQL)

## 1) Start MySQL

### XAMPP
1. Open **XAMPP Control Panel**
2. Click **Start** next to **MySQL**
3. Default is usually:
   - Host: `localhost`
   - Port: `3306`
   - User: `root`
   - Password: *(empty unless you set one)*

### MySQL installed as a Windows service
- Ensure **MySQL80** (or similar) is running in Services

## 2) Create the database

You must create the **database**. You do **not** need to create tables by hand.

### Option A — DBeaver
1. Open DBeaver → **Database → New Database Connection** → **MySQL**
2. Connect with:
   - Host: `localhost`
   - Port: `3306`
   - Username: `root`
   - Password: your MySQL password (or blank)
3. **Test Connection** → **Finish**
4. Right-click the connection → **SQL Editor → New SQL Script**
5. Run:

```sql
CREATE DATABASE IF NOT EXISTS voice_authentication;
```

6. Refresh the connection — you should see `voice_authentication`

### Option B — MySQL CLI / phpMyAdmin
```sql
CREATE DATABASE IF NOT EXISTS voice_authentication;
```

## 3) Configure `.env`

In `app/backend/`, create a file named `.env` (same folder as `package.json`):

```env
PORT=5000
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=voice_authentication
JWT_SECRET=replace_with_a_secure_secret
```

Notes:
- If root has **no password**, use `DB_PASSWORD=` (empty after `=`)
- `DB_USER` must not be empty — missing `.env` causes:  
  `Access denied for user ''@'localhost' (using password: NO)`
- Never commit `.env` (it is gitignored)

## 4) Install and run

```powershell
cd D:\speaker-verification-system\app\backend
npm install
npm run dev
```

On success you should see:

```text
Server running on http://localhost:5000
```

Check health:

```text
GET http://localhost:5000/api/health
```

### What gets auto-created
On first successful DB connection, the server creates:

| Table | Purpose |
|-------|---------|
| `users` | username + hashed password |
| `voice_samples` | enrollment/verification audio **paths** |

## Common errors

| Error | Fix |
|-------|-----|
| `'nodemon' is not recognized` | Run `npm install` in `app/backend` first |
| `Access denied for user ''@'localhost'` | Create `.env` with `DB_USER=root` (and password) |
| `Access denied for user 'root'@'localhost'` | Wrong password, or MySQL not allowing that user |
| `ECONNREFUSED` / cannot connect | MySQL is not running (start XAMPP MySQL) |
| `Unknown database 'voice_authentication'` | Run the `CREATE DATABASE` step |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server with nodemon |
| `npm start` | Production-style `node server.js` |

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/auth/register` | No | Register user |
| `POST` | `/api/auth/login` | No | Login → JWT |
| `GET` | `/api/users/profile` | Bearer | Current user profile |
| `POST` | `/api/voice/enroll` | Bearer + `audio` file | Save enrollment WAV |
| `POST` | `/api/voice/verify` | Bearer + `audio` file | Save verification WAV |
| `GET` | `/api/voice/history` | Bearer | Sample history |
| `GET` | `/api/health` | No | Health check |

CORS is configured for the Vite frontend at `http://localhost:5173`.

## Upload storage

Audio files are stored on **disk only** (not as BLOBs in MySQL):

```text
uploads/enrollments/user_<id>/enroll_YYYYMMDD_HHMMSS.wav
uploads/verifications/user_<id>/verify_YYYYMMDD_HHMMSS.wav
```

MySQL stores only the relative `file_path` in `voice_samples`.

## Project layout

```text
backend/
├── server.js
├── config/db.js          # MySQL pool + auto schema
├── controllers/
├── models/
├── routes/
├── middleware/
├── uploads/
├── .env                  # local only — create this
├── package.json
└── README.md
```
