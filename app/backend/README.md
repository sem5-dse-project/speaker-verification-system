# Voice Authentication Backend

## Setup

1. Create MySQL database:

```sql
CREATE DATABASE IF NOT EXISTS voice_authentication;
```

2. Configure environment variables in `.env`:

```env
PORT=5000
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=voice_authentication
JWT_SECRET=replace_with_a_secure_secret
```

3. Install dependencies and run:

```bash
npm install
npm run dev
```

The server auto-creates required tables (`users`, `voice_samples`) if they do not exist.

## API Endpoints

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/users/profile` (Bearer token)
- `POST /api/voice/enroll` (Bearer token + `audio` file)
- `POST /api/voice/verify` (Bearer token + `audio` file)
- `GET /api/voice/history` (Bearer token)

## Upload Storage

Audio files are stored on disk only (never in MySQL):

- `uploads/enrollments/user_<id>/enroll_YYYYMMDD_HHMMSS.wav`
- `uploads/verifications/user_<id>/verify_YYYYMMDD_HHMMSS.wav`

MySQL stores only relative file paths in `voice_samples.file_path`.
