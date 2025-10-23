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
      
      debugInfo.push(`Original image: ${originalWidth}x${originalHeight}`);
      
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
      
      // Convert to grayscale
      const gray = workingImg.bgrToGray();
      
      // Apply bilateral filter to reduce noise while keeping edges
      const filtered = gray.bilateralFilter(9, 75, 75);
      
      // Edge detection with Canny
      const edges = filtered.canny(50, 200);
      
      // Dilate to close gaps
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
      const dilated = edges.dilate(kernel);
      
      // Find contours
      const contours = dilated.findContours(
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE
      );
      
      debugInfo.push(`Found ${contours.length} contours`);
      
      // Sort by area (largest first)
      const sortedContours = contours.sort((a, b) => b.area - a.area);
      
      let processedImage = img;
      let transformed = false;
      const imageArea = workingImg.rows * workingImg.cols;
      
      // Try to find document contour
      for (let i = 0; i < Math.min(10, sortedContours.length); i++) {
        const contour = sortedContours[i];
        const contourArea = contour.area;
        const areaPercent = (contourArea / imageArea) * 100;
        
        // Skip if too small
        if (areaPercent < 15) {
          debugInfo.push(`Contour ${i}: ${areaPercent.toFixed(1)}% - too small, skipping`);
          continue;
        }
        
        const peri = contour.arcLength(true);
        
        // Try multiple epsilon values for approximation
        for (let epsilonFactor = 0.01; epsilonFactor <= 0.08; epsilonFactor += 0.01) {
          const approx = contour.approxPolyDP(epsilonFactor * peri, true);
          
          // Get points array - THIS IS THE FIX
          const points = approx.getDataAsArray();
          const numPoints = points.length;
          
          debugInfo.push(`  Contour ${i}, ε=${epsilonFactor.toFixed(2)}: ${numPoints} points, ${areaPercent.toFixed(1)}% area`);
          
          // We want exactly 4 corners (quadrilateral)
          if (numPoints === 4) {
            debugInfo.push(`  ✓ Found quadrilateral!`);
            
            // Extract corner coordinates
            let srcPoints = points.map(pt => {
              // Handle nested array format [[x, y]]
              if (Array.isArray(pt) && Array.isArray(pt[0])) {
                return [pt[0][0], pt[0][1]];
              }
              // Handle {x, y} object format
              if (pt.x !== undefined && pt.y !== undefined) {
                return [pt.x, pt.y];
              }
              // Already in [x, y] format
              return pt;
            });
            
            // Scale points back to original image size
            if (workingImg.cols !== originalWidth) {
              const scaleX = originalWidth / workingImg.cols;
              const scaleY = originalHeight / workingImg.rows;
              srcPoints = srcPoints.map(pt => [pt[0] * scaleX, pt[1] * scaleY]);
            }
            
            debugInfo.push(`  Source points: ${JSON.stringify(srcPoints)}`);
            
            // Order points: top-left, top-right, bottom-right, bottom-left
            const orderedPoints = orderPoints(srcPoints);
            
            // Calculate output dimensions (A4 at 300 DPI)
            const width = 2480;
            const height = 3508;
            
            const dstPoints = [
              [0, 0],
              [width - 1, 0],
              [width - 1, height - 1],
              [0, height - 1]
            ];
            
            try {
              // Perform perspective transform
              const M = cv.getPerspectiveTransform(
                orderedPoints.map(p => new cv.Point2(p[0], p[1])),
                dstPoints.map(p => new cv.Point2(p[0], p[1]))
              );
              
              processedImage = img.warpPerspective(M, new cv.Size(width, height));
              transformed = true;
              debugInfo.push('  ✓ Perspective transform successful!');
              break;
            } catch (e) {
              debugInfo.push(`  ✗ Transform failed: ${e.message}`);
            }
          }
        }
        
        if (transformed) break;
      }
      
      debugInfo.push(`Result: Transformed = ${transformed}`);
      
      // Fallback: smart crop if no transform
      if (!transformed) {
        debugInfo.push('Applying fallback crop (10% border removal)');
        const cropPercent = 0.10;
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
      
      // Enhance with Sharp (KEEP COLOR)
      const enhanced = await sharp(buffer)
        .resize(2480, 3508, { fit: 'inside' })
        .normalize()  // Improve contrast
        .sharpen({ sigma: 1.5 })  // Sharpen text
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
    
    // Return debug info in header
    res.setHeader('X-Debug-Info', JSON.stringify(debugInfo).substring(0, 8000)); // Limit header size
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=scan.pdf');
    res.send(Buffer.from(pdfBytes));
    
  } catch (error) {
    console.error('Error processing scan:', error);
    res.status(500).json({ 
      error: 'Failed to process images', 
      details: error.message
    });
  }
});

// Order points clockwise from top-left
function orderPoints(pts) {
  // Calculate centroid
  const centerX = pts.reduce((sum, pt) => sum + pt[0], 0) / 4;
  const centerY = pts.reduce((sum, pt) => sum + pt[1], 0) / 4;
  
  // Sort points by angle from center
  const sortedByAngle = pts.map(pt => ({
    point: pt,
    angle: Math.atan2(pt[1] - centerY, pt[0] - centerX)
  })).sort((a, b) => a.angle - b.angle);
  
  // Find top-left (has smallest x + y)
  let tlIndex = 0;
  let minSum = Infinity;
  for (let i = 0; i < 4; i++) {
    const sum = sortedByAngle[i].point[0] + sortedByAngle[i].point[1];
    if (sum < minSum) {
      minSum = sum;
      tlIndex = i;
    }
  }
  
  // Reorder clockwise starting from top-left
  const ordered = [];
  for (let i = 0; i < 4; i++) {
    ordered.push(sortedByAngle[(tlIndex + i) % 4].point);
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
