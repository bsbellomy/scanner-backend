FROM node:20-bullseye

# 1. Install dependencies for full OpenCV + Node build
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

# 2. Set workdir
WORKDIR /app

# 3. Copy package files
COPY package*.json ./

# 4. Install Node dependencies
RUN npm install

# 5. Build OpenCV itself (downloads & compiles C++ libs)
RUN npx opencv-build-npm rebuild --jobs max

# 6. Build the native Node bindings (the actual opencv4nodejs.node)
RUN npm rebuild @u4/opencv4nodejs --build-from-source

# 7. Verify native module exists
RUN test -f node_modules/@u4/opencv4nodejs/build/Release/opencv4nodejs.node || \
    (echo "❌ OpenCV node module build failed!" && exit 1)

# 8. Copy rest of app
COPY . .

# 9. Start command
CMD ["npm", "start"]
