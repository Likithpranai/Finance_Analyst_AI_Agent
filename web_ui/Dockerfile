FROM python:3.9-slim AS backend

WORKDIR /app

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy the main Finance Analyst AI Agent code
COPY ../finance_analyst_agent.py /app/
COPY ../tools/ /app/tools/
COPY ../utils/ /app/utils/
COPY ../models/ /app/models/
COPY ../config/ /app/config/
COPY ../data/ /app/data/

# Expose backend port
EXPOSE 8000

# Start backend server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend build stage
FROM node:16-alpine AS frontend-build

WORKDIR /app

# Copy frontend package.json and install dependencies
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

# Copy frontend code and build
COPY frontend/ .
RUN npm run build

# Production stage
FROM nginx:alpine AS production

# Copy built frontend assets from frontend-build stage
COPY --from=frontend-build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY web_ui/nginx.conf /etc/nginx/conf.d/default.conf

# Expose frontend port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
