FROM nginx:1.27-alpine

# Copy static frontend files into nginx html root
COPY index.html /usr/share/nginx/html/
COPY asteroid.html /usr/share/nginx/html/
COPY resources/ /usr/share/nginx/html/resources/
COPY models/ /usr/share/nginx/html/models/

# Nginx config with internal proxy to API service
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose HTTP
EXPOSE 80

# Nginx default CMD handles startup
