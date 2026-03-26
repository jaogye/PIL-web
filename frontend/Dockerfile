# ── Stage 1: Build React app ──────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json ./
RUN npm install --frozen-lockfile

COPY . .
RUN npm run build


# ── Stage 2: Serve via nginx ──────────────────────────────────────────────────
FROM nginx:1.27-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf.template

# Default values for Fly.io production; override in other environments.
ENV RESOLVER="[fdaa::3]"
ENV BACKEND_URL="http://lip2-api.internal:8000"

EXPOSE 80

# envsubst with explicit var list so nginx variables ($host, $uri, etc.) are left untouched.
CMD ["/bin/sh", "-c", "envsubst '${RESOLVER} ${BACKEND_URL}' < /etc/nginx/nginx.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
