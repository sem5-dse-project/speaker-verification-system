# Backend API Reference

Base URL (local): `http://localhost:5000`

All JSON responses use a `success` boolean. Protected routes need:

```http
Authorization: Bearer <token>
```

Get the token from `POST /api/auth/login`.

---

## Summary

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/health` | No | Health check |
| `POST` | `/api/auth/register` | No | Register user |
| `POST` | `/api/auth/login` | No | Login, returns JWT |
| `GET` | `/api/users/profile` | Bearer | Current user profile |
| `POST` | `/api/voice/enroll/reset` | Bearer | Clear enrollment samples + template |
| `POST` | `/api/voice/enroll` | Bearer | Upload enrollment WAV |
| `POST` | `/api/voice/verify` | Bearer | Upload verification WAV |
| `GET` | `/api/voice/verification-logs` | Bearer | List saved verify results |
| `GET` | `/api/voice/history` | Bearer | List uploaded samples |

---

## Health

### `GET /api/health`

No body.

**Response `200`**

```json
{
  "success": true,
  "message": "Voice authentication backend is running"
}
```

---

## Auth

### `POST /api/auth/register`

Create a new user.

**Headers**

```http
Content-Type: application/json
```

**Body**

```json
{
  "username": "alice",
  "password": "secret123"
}
```

**Response `201`**

```json
{
  "success": true,
  "message": "User registered successfully"
}
```

**Errors**

| Status | When |
|--------|------|
| `400` | Missing `username` or `password` |
| `409` | Username already exists |
| `500` | Server / DB error |

**curl**

```bash
curl -X POST http://localhost:5000/api/auth/register ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"alice\",\"password\":\"secret123\"}"
```

---

### `POST /api/auth/login`

Login and receive a JWT (expires in 1 day).

**Headers**

```http
Content-Type: application/json
```

**Body**

```json
{
  "username": "alice",
  "password": "secret123"
}
```

**Response `200`**

```json
{
  "success": true,
  "token": "<jwt>",
  "user": {
    "id": 1,
    "username": "alice"
  }
}
```

**Errors**

| Status | When |
|--------|------|
| `400` | Missing `username` or `password` |
| `401` | Invalid credentials |
| `500` | Server / DB error |

**curl**

```bash
curl -X POST http://localhost:5000/api/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"alice\",\"password\":\"secret123\"}"
```

---

## Users

### `GET /api/users/profile`

Return the authenticated user's profile.

**Headers**

```http
Authorization: Bearer <token>
```

**Response `200`**

```json
{
  "success": true,
  "user": {
    "id": 1,
    "username": "alice",
    "created_at": "2026-07-30T00:00:00.000Z"
  }
}
```

**Errors**

| Status | When |
|--------|------|
| `401` | Missing / invalid token |
| `404` | User not found |
| `500` | Server / DB error |

**curl**

```bash
curl http://localhost:5000/api/users/profile ^
  -H "Authorization: Bearer <token>"
```

---

## Voice

Audio must be **WAV**. Form field name must be **`audio`**. Max size: **20 MB**.

### `POST /api/voice/enroll/reset`

Deletes all enrollment samples for the user (and their files) and clears `enrollment_templates`. The frontend calls this before uploading a fresh set of 3 samples.

**Response `200`**

```json
{
  "success": true,
  "message": "Enrollment samples and template cleared",
  "deleted_samples": 3
}
```

### `POST /api/voice/enroll`

Upload an enrollment sample. File is saved under `uploads/enrollments/user_<id>/`.

**Headers**

```http
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Form fields**

| Field | Type | Required |
|-------|------|----------|
| `audio` | file (`.wav`) | Yes |

**Response `201`** (after 1st/2nd sample — template still pending)

```json
{
  "success": true,
  "message": "Enrollment audio uploaded (1/3 samples)",
  "sample": {
    "id": 1,
    "user_id": 1,
    "file_path": "uploads/enrollments/user_1/enroll_u1_s1_20260730_001530142_a3f2c1.wav",
    "sample_type": "enrollment"
  },
  "enrollment_count": 1,
  "required_samples": 3,
  "template_status": "pending",
  "template": null
}
```

After the **3rd** enrollment sample (with Python ML server running), `template_status` becomes `"ready"` and MySQL stores the averaged embedding in `enrollment_templates`.

**Errors**

| Status | When |
|--------|------|
| `400` | No file, or not a WAV |
| `401` | Missing / invalid token |
| `500` | Upload / DB error |

**curl**

```bash
curl -X POST http://localhost:5000/api/voice/enroll ^
  -H "Authorization: Bearer <token>" ^
  -F "audio=@D:\path\to\sample.wav"
```

---

### `POST /api/voice/verify`

Upload a verification sample. File is saved under `uploads/verifications/user_<id>/`.

> Requires a stored enrollment template (built after 3 enrollment uploads) and the Python ML server on `ML_SERVER_URL`.
> With `REPLAY_DETECTION=true` (default), Express calls `/replay/detect` first. Replay → `decision: "REPLAY"` (speaker verify skipped).

**Headers**

```http
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Form fields**

| Field | Type | Required |
|-------|------|----------|
| `audio` | file (`.wav`) | Yes |

**Response `201`** (live speaker match)

```json
{
  "success": true,
  "message": "Verification complete",
  "sample": {
    "id": 2,
    "user_id": 1,
    "file_path": "uploads/verifications/user_1/verify_20260730_001600.wav",
    "sample_type": "verification"
  },
  "replay": {
    "score": 0.12,
    "threshold": 0.765,
    "is_replay": false,
    "accepted": true,
    "decision": "LIVE"
  },
  "result": {
    "score": 0.72,
    "threshold": 0.25,
    "accepted": true,
    "decision": "ACCEPT"
  },
  "log": {
    "id": 1,
    "user_id": 1,
    "voice_sample_id": 2,
    "score": 0.72,
    "threshold": 0.25,
    "accepted": true,
    "decision": "ACCEPT",
    "created_at": "2026-07-30T00:16:00.000Z"
  }
}
```

**Response `201`** (replay blocked)

```json
{
  "success": true,
  "message": "Verification rejected: replay attack detected",
  "replay": {
    "score": 0.91,
    "threshold": 0.765,
    "is_replay": true,
    "decision": "REPLAY"
  },
  "result": {
    "score": 0.91,
    "threshold": 0.765,
    "accepted": false,
    "decision": "REPLAY"
  }
}
```

The `log` row is also stored in MySQL `verification_logs` (`decision` may be `ACCEPT`, `REJECT`, or `REPLAY`).

**Errors**

| Status | When |
|--------|------|
| `400` | No file, or not a WAV |
| `401` | Missing / invalid token |
| `500` | Upload / DB error |

**curl**

```bash
curl -X POST http://localhost:5000/api/voice/verify ^
  -H "Authorization: Bearer <token>" ^
  -F "audio=@D:\path\to\probe.wav"
```

---

### `GET /api/voice/verification-logs`

List saved verification results for the current user (newest first). Optional query: `?limit=50`.

**Headers**

```http
Authorization: Bearer <token>
```

**Response `200`**

```json
{
  "success": true,
  "logs": [
    {
      "id": 1,
      "user_id": 1,
      "voice_sample_id": 2,
      "file_path": "uploads/verifications/user_1/verify_u1_s1_20260730_001600142_a3f2c1.wav",
      "score": 0.72,
      "threshold": 0.25,
      "accepted": true,
      "decision": "ACCEPT",
      "created_at": "2026-07-30T00:16:00.000Z"
    }
  ]
}
```

**Errors**

| Status | When |
|--------|------|
| `401` | Missing / invalid token |
| `500` | Server / DB error |

**curl**

```bash
curl "http://localhost:5000/api/voice/verification-logs?limit=20" ^
  -H "Authorization: Bearer <token>"
```

---

### `GET /api/voice/history`

List enrollment and verification samples for the current user (newest first).

**Headers**

```http
Authorization: Bearer <token>
```

**Response `200`**

```json
{
  "success": true,
  "history": [
    {
      "id": 2,
      "user_id": 1,
      "file_path": "uploads/verifications/user_1/verify_20260730_001600.wav",
      "sample_type": "verification",
      "created_at": "2026-07-30T00:16:00.000Z"
    },
    {
      "id": 1,
      "user_id": 1,
      "file_path": "uploads/enrollments/user_1/enroll_u1_s1_20260730_001530142_a3f2c1.wav",
      "sample_type": "enrollment",
      "created_at": "2026-07-30T00:15:30.000Z"
    }
  ],
  "verification_logs": [
    {
      "id": 1,
      "user_id": 1,
      "voice_sample_id": 2,
      "file_path": "uploads/verifications/user_1/verify_u1_s1_20260730_001600142_a3f2c1.wav",
      "score": 0.72,
      "threshold": 0.25,
      "accepted": true,
      "decision": "ACCEPT",
      "created_at": "2026-07-30T00:16:00.000Z"
    }
  ]
}
```

**Errors**

| Status | When |
|--------|------|
| `401` | Missing / invalid token |
| `500` | Server / DB error |

**curl**

```bash
curl http://localhost:5000/api/voice/history ^
  -H "Authorization: Bearer <token>"
```

---

## Auth header example (Postman)

1. `POST /api/auth/login` → copy `token`
2. For protected routes: **Authorization** type **Bearer Token** → paste token
3. For enroll/verify: Body → **form-data** → key `audio` (type File) → choose `.wav`

---

## Static uploads

Uploaded files are also served statically:

```text
GET http://localhost:5000/uploads/enrollments/user_1/enroll_....wav
```

(path must match `file_path` returned by the API)
