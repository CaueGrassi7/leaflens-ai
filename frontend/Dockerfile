# Frontend Dockerfile for LeafLens AI
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Accept build argument for API URL (defaults to localhost for development)
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Build the Next.js application
# Next.js will prefer next.config.js over next.config.ts if both exist
RUN npm run build

# Stage 2: Runner
FROM node:20-alpine AS runner

WORKDIR /app

# Set to production environment
ENV NODE_ENV=production

# Copy necessary files from builder
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
# Copy next.config.js (Next.js will prefer .js over .ts if both exist)
COPY --from=builder /app/next.config.js ./

# Install production dependencies only
RUN npm ci --only=production

# Expose port
EXPOSE 3000

# Start the Next.js server
CMD ["npm", "run", "start"]

