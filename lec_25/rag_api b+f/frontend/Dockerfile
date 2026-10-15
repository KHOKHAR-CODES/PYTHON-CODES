FROM python:3.13-slim
WORKDIR /usr/share/nginx/html
COPY index.html script.js style.css ./
EXPOSE 3000
# Serve via simple Python HTTP server on port 3000 for frontend
CMD ["python", "-m", "http.server", "3000", "--bind", "0.0.0.0"]
