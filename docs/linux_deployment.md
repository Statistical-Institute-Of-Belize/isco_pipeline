# ISCO Pipeline Deployment on AlmaLinux

## 1. Install Docker on AlmaLinux

```bash
# Update your system
sudo dnf update -y

# Install utilities needed for Docker repository
sudo dnf install -y dnf-utils

# Add the Docker repository
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker packages
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify Docker is installed correctly
sudo docker --version
```

## 2. Install Docker Compose

```bash
# Download Docker Compose binary
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make it executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify Docker Compose installation
docker-compose --version
```

## 3. Add Your User to Docker Group (Optional, for security)

```bash
# Add your user to the docker group to run docker without sudo
sudo usermod -aG docker $USER

# Apply the new group (log out and back in, or run this)
newgrp docker
```

## 4. Load the Docker Image

```bash
# Assuming you've copied isco-pipeline.tar to the server
docker load < isco-pipeline.tar

# Verify the image is loaded
docker images
```

## 5. Prepare Environment for ISCO Pipeline

```bash
# Create a directory for the application
mkdir -p ~/isco-pipeline
cd ~/isco-pipeline

# Create required directories
mkdir -p data/raw data/processed data/reference models/best_model logs
```

## 6. Create Docker Compose File

Create a file named `docker-compose.yml` with the following content:

```yaml
version: '3'

services:
  isco-api:
    image: isco-pipeline:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models/best_model:/app/models/best_model
      - ./logs:/app/logs
    command: api
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

## 7. Deploy and Manage the Application

```bash
# Start the application
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

## 8. Use the API

The API is now accessible at `http://your-server-ip:8000/docs`

## 9. Use the CLI Mode (if needed)

```bash
# Run CLI commands using Docker
docker-compose run --rm isco-api cli --input data/new/my_data.csv
```

## 10. Updating the Application

When you have a new version of the application:

```bash
# Load the new image
docker load < new-isco-pipeline.tar

# Restart the service to use the new image
docker-compose down
docker-compose up -d
```

## 11. Firewall Configuration (if needed)

If you're using firewalld:

```bash
# Open port 8000 for API access
sudo firewall-cmd --zone=public --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

## 12. SSL Configuration (for production)

For production environments, consider using a reverse proxy like Nginx with SSL:

```bash
sudo dnf install -y nginx certbot python3-certbot-nginx
```

Then configure Nginx to proxy requests to your Docker container.

## Troubleshooting

- **Container won't start**: Check logs with `docker-compose logs`
- **Can't connect to API**: Verify ports are open with `sudo firewall-cmd --list-all`
- **Missing model files**: Ensure model files are correctly placed in `models/best_model/`