# Use Node 20 base image
FROM node:20-bullseye

# Install all build dependencies and OpenCV requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency manifests
COPY package*.json ./

# Install Node dependencies (includes @u4/opencv4nodejs)
RUN npm install

# Force OpenCV build using opencv-build-npm
RUN npx opencv-build-npm rebuild --jobs max

# ✅ Verify native module exists after build
RUN ls -l node_modules/@u4/opencv4nodejs/build/Release || (echo "opencv4nodejs build folder missing!" && exit 1)
RUN test -f node_modules/@u4/opencv4nodejs/build/Release/opencv4nodejs.node || (echo "OpenCV build failed!" && exit 1)

# Copy rest of app
COPY . .

# Start your app
CMD ["npm", "start"]
