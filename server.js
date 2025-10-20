const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const { PDFDocument } = require('pdf-lib');
const cv = require('@u4/opencv4nodejs');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.post('/process-scan', upload.array('images'), async (req, res) => {
  try {
    const pdfDoc = await PDFDocument.create();
    
    for (let file of req.files) {
      // Decode image with OpenCV
      const img = cv.imdecode(file.buffer);
      
      // Convert to grayscale
      const gray = img.cvtColor(cv.COLOR_BGR2GRAY);
      
      // Gaussian blur to reduce noise
      const blurred = gray.gaussianBlur(new cv.Size(5, 5), 0);
      
      // Edge detection
      const edges = blurred.canny(50, 150);
      
      // Find contours
      const contours = edges.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      
      // Get largest contour (should be document)
      const sortedContours = contours.sort((a, b) => b.area - a.area);
      let processedImage;
      
      if (sortedContours.length > 0) {
        const largestContour = sortedContours[0];
        const approx = largestContour.approxPolyDP(0.02 * largestContour.arcLength(true), true);
        
        // If we found a 4-point contour (rectangle), do perspective transform
        if (approx.rows === 4) {
          const srcPoints = approx.getDataAsArray().map(pt => pt[0]);
          const width = 2480;  // A4 at 300 DPI
          const height = 3508;
          
          const dstPoints = [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
          ];
          
          const M = cv.getPerspectiveTransform(srcPoints, dstPoints);
          processedImage = img.warpPerspective(M, new cv.Size(width, height));
        } else {
          processedImage = img;
        }
      } else {
        processedImage = img;
      }
      
      // Convert back to buffer for Sharp enhancement
      const buffer = cv.imencode('.jpg', processedImage);
      
      // Enhance with Sharp
      const enhanced = await sharp(buffer)
        .grayscale()
        .normalize()
        .sharpen()
        .toBuffer();
      
      // Add to PDF
      const pdfImage = await pdfDoc.embedJpg(enhanced);
      const page = pdfDoc.addPage([2480, 3508]);
      page.drawImage(pdfImage, {
        x: 0,
        y: 0,
        width: 2480,
        height: 3508
      });
    }
    
    const pdfBytes = await pdfDoc.save();
    
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=scan.pdf');
    res.send(Buffer.from(pdfBytes));
    
  } catch (error) {
    console.error('Error processing scan:', error);
    res.status(500).json({ error: 'Failed to process images' });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', opencv: cv.version });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Scanner backend running on port ${PORT}`);
  console.log(`OpenCV version: ${cv.version.major}.${cv.version.minor}.${cv.version.revision}`);
});