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

# Install dependencies
RUN npm install

# Build OpenCV - with explicit output directory
ENV OPENCV4NODEJS_AUTOBUILD_OPENCV_VERSION=4.5.5
ENV OPENCV4NODEJS_DISABLE_AUTOBUILD=0
RUN npx opencv-build-npm rebuild

# Verify build succeeded
RUN test -f /app/node_modules/@u4/opencv4nodejs/build/Release/opencv4nodejs.node || \
    (echo "OpenCV build failed!" && exit 1)

# Copy application code
COPY . .

EXPOSE 3000

CMD ["npm", "start"]
