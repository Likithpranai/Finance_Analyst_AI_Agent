FROM node:16-alpine

WORKDIR /app

# Copy package.json and install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy frontend code
COPY . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]
