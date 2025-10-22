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
    
    for (let file of req.files) {
      // Decode image with OpenCV
      let img = cv.imdecode(file.buffer);
      
      // Convert to grayscale
      const gray = img.bgrToGray();
      
      // Apply Gaussian blur
      const blurred = gray.gaussianBlur(new cv.Size(5, 5), 0);
      
      // Adaptive threshold for better edge detection
      const thresh = blurred.adaptiveThreshold(
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,
        2
      );
      
      // Find contours
      const contours = thresh.findContours(
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
      );
      
      // Sort by area and get largest
      const sortedContours = contours.sort((a, b) => b.area - a.area);
      
      let processedImage = img;
      let transformed = false;
      
      // Try to find a quadrilateral in the largest contours
      for (let i = 0; i < Math.min(5, sortedContours.length); i++) {
        const contour = sortedContours[i];
        const peri = contour.arcLength(true);
        const approx = contour.approxPolyDP(0.02 * peri, true);
        
        // Check if it's roughly the size of the whole image (document)
        const imageArea = img.rows * img.cols;
        const contourArea = contour.area;
        
        if (approx.rows === 4 && contourArea > imageArea * 0.3) {
          // Found a good quadrilateral!
          const srcPoints = approx.getDataAsArray().map(pt => pt[0]);
          
          // Order points: top-left, top-right, bottom-right, bottom-left
          const orderedPoints = orderPoints(srcPoints);
          
          // Calculate destination dimensions
          const width = 2480;  // A4 at 300 DPI
          const height = 3508;
          
          const dstPoints = [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
          ];
          
          // Perspective transform
          const M = cv.getPerspectiveTransform(orderedPoints, dstPoints);
          processedImage = img.warpPerspective(M, new cv.Size(width, height));
          transformed = true;
          break;
        }
      }
      
      // If no transformation happened, at least crop some borders
      if (!transformed) {
        const cropPercent = 0.05; // Remove 5% from each edge
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
      
      // Enhance with Sharp
      const enhanced = await sharp(buffer)
        .resize(2480, 3508, { fit: 'inside' })
        .grayscale()
        .normalize()
        .sharpen()
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
    
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=scan.pdf');
    res.send(Buffer.from(pdfBytes));
    
  } catch (error) {
    console.error('Error processing scan:', error);
    res.status(500).json({ error: 'Failed to process images', details: error.message });
  }
});

// Helper function to order points
function orderPoints(pts) {
  // Sort by sum (top-left has smallest sum, bottom-right has largest)
  const sorted = pts.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]));
  
  const tl = sorted[0];
  const br = sorted[3];
  
  // Sort remaining by difference
  const remaining = [sorted[1], sorted[2]];
  remaining.sort((a, b) => (a[1] - a[0]) - (b[1] - b[0]));
  
  const tr = remaining[1];
  const bl = remaining[0];
  
  return [tl, tr, br, bl];
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
