# Docker Quick Start Guide

Get MailFind running with Docker in under 5 minutes!

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Your environment variables ready (see below)

## Quick Start

### 1. Clone and Navigate

```bash
cd /path/to/mailfind
```

### 2. Create Environment File

```bash
# Copy the example file (if it exists) or create .env manually
cp .env.example .env
```

Edit `.env` with your actual values:

```env
# Supabase Configuration
SUPABASE_PUBLIC_URL=https://your-project.supabase.co
SERVICE_ROLE=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key

# Encryption Configuration
# Generate using: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=your-fernet-encryption-key
```

### 3. Start Services

**Option A: Using the helper script (recommended)**

```bash
./scripts/local-docker-start.sh
```

**Option B: Using docker-compose directly**

```bash
docker-compose up -d
```

### 4. Verify Services

```bash
# Check if services are running
docker-compose ps

# Check backend health
curl http://localhost:8000/

# View logs
docker-compose logs -f
```

### 5. Access the Application

- **Backend API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs (if FastAPI docs are enabled)

### 6. Authenticate with Google

Navigate to http://localhost:8000/login in your browser and complete the OAuth flow.

## Common Commands

### Start Services

```bash
# Start in background
docker-compose up -d

# Start with rebuild
docker-compose up -d --build

# Start in foreground (see logs)
docker-compose up
```

### View Logs

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100
```

### Stop Services

```bash
# Stop services (keep containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers, and remove volumes
docker-compose down -v
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart backend only
docker-compose restart backend
```

### Execute Commands in Containers

```bash
# Access backend shell
docker-compose exec backend /bin/bash

# Run Python in backend
docker-compose exec backend python

# Check backend Python packages
docker-compose exec backend pip list
```

### Update and Rebuild

```bash
# Rebuild images
docker-compose build

# Rebuild and restart
docker-compose up -d --build

# Force recreation of containers
docker-compose up -d --force-recreate
```

## Development Workflow

### Code Changes

The `docker-compose.yml` mounts your local code as volumes, so changes are reflected immediately (with hot reload for development mode).

To enable development mode in backend:

```yaml
# In docker-compose.yml, add to backend service:
command: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Rebuild After Dependency Changes

If you update `requirements.txt` or `package.json`:

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Container Fails to Start

```bash
# Check logs
docker-compose logs backend

# Check container status
docker-compose ps

# Inspect container
docker inspect mailfind-backend
```

### Environment Variables Not Loading

```bash
# Verify .env file exists
ls -la .env

# Check if variables are set in container
docker-compose exec backend env | grep SUPABASE
```

### Database Connection Issues

1. Verify SUPABASE_PUBLIC_URL is correct
2. Check SERVICE_ROLE key is valid
3. Ensure network allows outbound HTTPS

```bash
# Test connection from container
docker-compose exec backend curl https://your-project.supabase.co
```

### Clean Start

If things are really broken, start fresh:

```bash
# Stop everything
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Remove all Docker resources (careful!)
docker system prune -a --volumes

# Rebuild and start
docker-compose build
docker-compose up -d
```

## Production Deployment

For production deployment with Kubernetes, see:
- [DOCKER_KUBERNETES_GUIDE.md](./DOCKER_KUBERNETES_GUIDE.md)

## Next Steps

1. ✅ Services running
2. ✅ Authenticated with Google
3. Load the Chrome extension (see main README.md)
4. Start searching your emails!

## Support

For more detailed information:
- [Main README](./README.md)
- [Docker & Kubernetes Guide](./DOCKER_KUBERNETES_GUIDE.md)
- [Backend Documentation](./packages/backend/README.md)

