FROM node:20-bullseye

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node dependencies and build OpenCV
RUN npm install && \
    npx opencv-build-npm rebuild

# Copy application code
COPY . .

EXPOSE 3000

CMD ["npm", "start"]
