# Voice Authentication Frontend (React + Vite)

Web UI for the voice authentication system. Users can register/login, open a dashboard, enroll voice samples, and submit verification audio.

Talks to the Express backend at `http://localhost:5000/api` (see `app/backend`).

> **Note:** The recorder captures PCM via Web Audio and encodes real 16-bit WAV files (not browser WebM). Enrollment resets previous samples before uploading a new set of 3.

## Features

| Page | Route | Description |
|------|-------|-------------|
| Login | `/login` | Sign in → JWT saved in `localStorage` |
| Register | `/register` | Create account |
| Dashboard | `/dashboard` | Home after login (protected) |
| Enrollment | `/enrollment` | Record **3** enrollment samples, then upload (protected) |
| Verification | `/verification` | Record/upload verification audio (protected) |

Stack: **React 19**, **Vite 8**, **React Router**, **Axios**, **Tailwind CSS 4**, **Lucide** icons.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server (HMR) |
| `npm run build` | Production build → `dist/` |
| `npm run preview` | Preview production build |
| `npm run lint` | Run Oxlint |
| `npm test` | Run unit tests (Vitest) |

## Unit tests

```powershell
cd D:\speaker-verification-system\app\frontend
npm test
```

Tests cover WAV encoding, auth token checks, password strength, API interceptor, and core UI helpers.

## Prerequisites

- **Node.js** 18+ (npm included)
- Backend running (MySQL + Express):

```powershell
cd D:\speaker-verification-system\app\backend
npm install
npm run dev
```

Backend should show: `Server running on http://localhost:5000`

Setup details: [`../backend/README.md`](../backend/README.md)  
API docs: [`../backend/API.md`](../backend/API.md)

## Setup

```powershell
cd D:\speaker-verification-system\app\frontend
npm install
npm run dev
```

Vite will print a local URL, usually:

```text
http://localhost:5173
```

Open that in the browser. The backend CORS config already allows `http://localhost:5173`.

## How it connects to the backend

`src/services/api.js` uses Axios with:

```text
baseURL = http://localhost:5000/api
```

On each request, if `localStorage.token` exists, it sends:

```http
Authorization: Bearer <token>
```

Typical flow:

1. Register / Login → save `token`
2. Protected pages use `ProtectedRoute`
3. Enrollment / Verification upload audio to `/api/voice/enroll` or `/api/voice/verify`

## Project layout

```text
frontend/
├── src/
│   ├── App.jsx                 # Routes
│   ├── main.jsx
│   ├── pages/
│   │   ├── Login.jsx
│   │   ├── Register.jsx
│   │   ├── Dashboard.jsx
│   │   ├── Enrollment.jsx
│   │   └── Verification.jsx
│   ├── components/             # UI helpers (Recorder, AuthCard, …)
│   └── services/
│       └── api.js              # Axios client → Express API
├── index.html
├── vite.config.js
├── package.json
└── README.md
```

## Common issues

| Problem | Fix |
|---------|-----|
| Login/register network error | Start backend on port **5000** |
| CORS error | Use frontend on `http://localhost:5173` (not another origin) |
| Redirected to login on dashboard | Token missing/expired — log in again |
| Voice upload fails | Backend expects **WAV**; field name `audio` |
| `'vite' is not recognized` | Run `npm install` in `app/frontend` first |

## Related

- Backend setup: `app/backend/README.md`
- Backend API: `app/backend/API.md`
