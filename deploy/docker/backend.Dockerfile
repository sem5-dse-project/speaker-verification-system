# Build: docker build -f deploy/docker/backend.Dockerfile -t voice-auth-backend .
FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/backend/package.json app/backend/package-lock.json ./
RUN npm ci --omit=dev

COPY app/backend/ ./

RUN mkdir -p uploads

ENV NODE_ENV=production
ENV PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -sf http://127.0.0.1:5000/api/health || exit 1

CMD ["node", "server.js"]
