FROM node:20-bookworm-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ENV HOST=0.0.0.0 \
    PORT=3000 \
    BROWSER=none

EXPOSE 3000

CMD ["npm", "start"]
