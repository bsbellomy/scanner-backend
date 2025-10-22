const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const { PDFDocument } = require('pdf-lib');
const cv = require('@u4/opencv4nodejs');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.post('/process-scan', upload.array('images'), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No images provided' });
    }

    const pdfDoc = await PDFDocument.create();
    const debugInfo = [];
    
    for (let fileIndex = 0; fileIndex < req.files.length; fileIndex++) {
      const file = req.files[fileIndex];
      
      // Decode image with OpenCV
      let img = cv.imdecode(file.buffer);
      const originalHeight = img.rows;
      const originalWidth = img.cols;
      
      // Resize if too large (for faster processing)
      let workingImg = img;
      const maxDimension = 1500;
      if (img.cols > maxDimension || img.rows > maxDimension) {
        const scale = maxDimension / Math.max(img.cols, img.rows);
        workingImg = img.resize(
          Math.floor(img.rows * scale),
          Math.floor(img.cols * scale)
        );
      }
      
      // Convert to grayscale for edge detection
      const gray = workingImg.bgrToGray();
      
      // Apply bilateral filter to reduce noise while keeping edges sharp
      const filtered = gray.bilateralFilter(9, 75, 75);
      
      // Edge detection using Canny
      const edges = filtered.canny(30, 100);
      
      // Dilate to close gaps in edges
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
      const dilated = edges.dilate(kernel);
      
      // Find contours
      const contours = dilated.findContours(
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
      );
      
      debugInfo.push(`Found ${contours.length} contours`);
      
      // Sort by area
      const sortedContours = contours.sort((a, b) => b.area - a.area);
      
      let processedImage = img;
      let transformed = false;
      const imageArea = workingImg.rows * workingImg.cols;
      
      // Try to find document contour (much more lenient)
      for (let i = 0; i < Math.min(10, sortedContours.length); i++) {
        const contour = sortedContours[i];
        const peri = contour.arcLength(true);
        
        // Try different approximation values
        for (let epsilon = 0.01; epsilon <= 0.05; epsilon += 0.01) {
          const approx = contour.approxPolyDP(epsilon * peri, true);
          const contourArea = contour.area;
          const areaPercent = (contourArea / imageArea) * 100;
          
          debugInfo.push(`Contour ${i}, epsilon ${epsilon.toFixed(2)}: ${approx.rows} points, ${areaPercent.toFixed(1)}% of image`);
          
          // Accept if it's a quadrilateral and takes up significant space (lowered to 20%)
          if (approx.rows === 4 && areaPercent > 20) {
            debugInfo.push(`✓ Using this contour!`);
            
            // Get corner points
            let srcPoints = approx.getDataAsArray().map(pt => pt[0]);
            
            // Scale points back to original image size if we resized
            if (workingImg.cols !== originalWidth) {
              const scaleX = originalWidth / workingImg.cols;
              const scaleY = originalHeight / workingImg.rows;
              srcPoints = srcPoints.map(pt => [pt[0] * scaleX, pt[1] * scaleY]);
            }
            
            // Order points: top-left, top-right, bottom-right, bottom-left
            const orderedPoints = orderPoints(srcPoints);
            
            // Calculate dimensions for output
            const width = 2480;  // A4 at 300 DPI
            const height = 3508;
            
            const dstPoints = [
              [0, 0],
              [width - 1, 0],
              [width - 1, height - 1],
              [0, height - 1]
            ];
            
            try {
              // Perspective transform
              const M = cv.getPerspectiveTransform(orderedPoints, dstPoints);
              processedImage = img.warpPerspective(M, new cv.Size(width, height));
              transformed = true;
              debugInfo.push('Transform successful!');
            } catch (e) {
              debugInfo.push(`Transform failed: ${e.message}`);
            }
            
            if (transformed) break;
          }
        }
        
        if (transformed) break;
      }
      
      debugInfo.push(`Final result: Transformed = ${transformed}`);
      
      // If no transformation, do smart cropping
      if (!transformed) {
        debugInfo.push('No valid quadrilateral found - applying smart crop');
        
        // Crop 8% from each edge
        const cropPercent = 0.08;
        const cropX = Math.floor(img.cols * cropPercent);
        const cropY = Math.floor(img.rows * cropPercent);
        const cropWidth = Math.floor(img.cols * (1 - 2 * cropPercent));
        const cropHeight = Math.floor(img.rows * (1 - 2 * cropPercent));
        
        processedImage = img.getRegion(
          new cv.Rect(cropX, cropY, cropWidth, cropHeight)
        );
      }
      
      // Convert to buffer
      const buffer = cv.imencode('.jpg', processedImage);
      
      // Enhance with Sharp (keep color)
      const enhanced = await sharp(buffer)
        .resize(2480, 3508, { fit: 'inside' })
        .normalize()  // Improve contrast
        .sharpen()     // Sharpen text
        .toBuffer();
      
      // Add to PDF
      const pdfImage = await pdfDoc.embedJpg(enhanced);
      const page = pdfDoc.addPage([2480, 3508]);
      const { width: imgWidth, height: imgHeight } = pdfImage.scale(1);
      
      page.drawImage(pdfImage, {
        x: (2480 - imgWidth) / 2,
        y: (3508 - imgHeight) / 2,
        width: imgWidth,
        height: imgHeight
      });
    }
    
    const pdfBytes = await pdfDoc.save();
    
    // Add debug info to response headers
    res.setHeader('X-Debug-Info', JSON.stringify(debugInfo));
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=scan.pdf');
    res.send(Buffer.from(pdfBytes));
    
  } catch (error) {
    console.error('Error processing scan:', error);
    res.status(500).json({ 
      error: 'Failed to process images', 
      details: error.message,
      stack: error.stack
    });
  }
});

// Helper function to order points clockwise from top-left
function orderPoints(pts) {
  // Calculate center
  const centerX = pts.reduce((sum, pt) => sum + pt[0], 0) / 4;
  const centerY = pts.reduce((sum, pt) => sum + pt[1], 0) / 4;
  
  // Sort by angle from center
  const angles = pts.map(pt => ({
    point: pt,
    angle: Math.atan2(pt[1] - centerY, pt[0] - centerX)
  }));
  
  angles.sort((a, b) => a.angle - b.angle);
  
  // Find top-left (smallest x+y sum)
  let minIdx = 0;
  let minSum = Infinity;
  for (let i = 0; i < 4; i++) {
    const sum = angles[i].point[0] + angles[i].point[1];
    if (sum < minSum) {
      minSum = sum;
      minIdx = i;
    }
  }
  
  // Reorder starting from top-left, going clockwise
  const ordered = [];
  for (let i = 0; i < 4; i++) {
    ordered.push(angles[(minIdx + i) % 4].point);
  }
  
  return ordered;
}

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    opencv: cv.version 
  });
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Scanner backend running on port ${PORT}`);
  console.log(`OpenCV version: ${cv.version.major}.${cv.version.minor}.${cv.version.revision}`);
});
