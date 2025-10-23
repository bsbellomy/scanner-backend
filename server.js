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
    
    for (let file of req.files) {
      // Decode image
      let img = cv.imdecode(file.buffer);
      const originalHeight = img.rows;
      const originalWidth = img.cols;
      
      debugInfo.push(`Original: ${originalWidth}x${originalHeight}`);
      
      // Resize for processing
      let workingImg = img;
      const maxDim = 1200;
      if (img.cols > maxDim || img.rows > maxDim) {
        const scale = maxDim / Math.max(img.cols, img.rows);
        workingImg = img.resize(
          Math.floor(img.rows * scale),
          Math.floor(img.cols * scale)
        );
      }
      
      // Convert to grayscale
      const gray = workingImg.bgrToGray();
      
      // Apply Gaussian blur
      const blurred = gray.gaussianBlur(new cv.Size(5, 5), 0);
      
      // ADAPTIVE THRESHOLDING - finds light areas (document) vs dark (background)
      const thresh = blurred.adaptiveThreshold(
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        115,  // Large block size to ignore texture
        10
      );
      
      // Morphological closing to fill gaps
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(15, 15));
      const closed = thresh.morphologyEx(kernel, cv.MORPH_CLOSE);
      
      // Find contours
      const contours = closed.findContours(
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
      );
      
      debugInfo.push(`Found ${contours.length} contours`);
      
      // Sort by area
      const sorted = contours.sort((a, b) => b.area - a.area);
      
      let processedImage = img;
      let transformed = false;
      const imageArea = workingImg.rows * workingImg.cols;
      
      // Look for large rectangular contours
      for (let i = 0; i < Math.min(5, sorted.length); i++) {
        const contour = sorted[i];
        const area = contour.area;
        const areaPercent = (area / imageArea) * 100;
        
        debugInfo.push(`Contour ${i}: ${areaPercent.toFixed(1)}% of image`);
        
        if (areaPercent < 40) {
          debugInfo.push(`  Too small, skipping`);
          continue;
        }
        
        // Approximate to polygon
        const peri = contour.arcLength(true);
        const approx = contour.approxPolyDP(0.02 * peri, true);
        
        // Extract points - FIXED METHOD
        let srcPoints = [];
        try {
          // Method 1: Try accessing as NumericArray
          for (let j = 0; j < approx.sizes[0]; j++) {
            const x = approx.at(j, 0).x;
            const y = approx.at(j, 0).y;
            srcPoints.push([x, y]);
          }
        } catch (e1) {
          try {
            // Method 2: Direct iteration
            for (let j = 0; j < approx.rows; j++) {
              srcPoints.push([approx.at(j).x, approx.at(j).y]);
            }
          } catch (e2) {
            debugInfo.push(`  Failed to extract points: ${e2.message}`);
            continue;
          }
        }
        
        const numPoints = srcPoints.length;
        debugInfo.push(`  ${numPoints} corners`);
        
        if (numPoints === 4) {
          debugInfo.push(`  ✓ Quadrilateral found!`);
          
          // Scale back to original size
          if (workingImg.cols !== originalWidth) {
            const scaleX = originalWidth / workingImg.cols;
            const scaleY = originalHeight / workingImg.rows;
            srcPoints = srcPoints.map(p => [p[0] * scaleX, p[1] * scaleY]);
          }
          
          // Order corners
          const ordered = orderPoints(srcPoints);
          debugInfo.push(`  Corners: ${JSON.stringify(ordered.map(p => p.map(n => Math.round(n))))}`);
          
          // Destination: A4 300dpi
          const dstWidth = 2480;
          const dstHeight = 3508;
          const dst = [
            [0, 0],
            [dstWidth - 1, 0],
            [dstWidth - 1, dstHeight - 1],
            [0, dstHeight - 1]
          ];
          
          try {
            const M = cv.getPerspectiveTransform(
              ordered.map(p => new cv.Point2(p[0], p[1])),
              dst.map(p => new cv.Point2(p[0], p[1]))
            );
            
            processedImage = img.warpPerspective(M, new cv.Size(dstWidth, dstHeight));
            transformed = true;
            debugInfo.push(`  ✓ Transform successful!`);
            break;
          } catch (e) {
            debugInfo.push(`  ✗ Transform error: ${e.message}`);
          }
        }
      }
      
      debugInfo.push(`Result: ${transformed ? 'TRANSFORMED' : 'FALLBACK CROP'}`);
      
      // Fallback
      if (!transformed) {
        const crop = 0.03;  // Only 3% crop
        const x = Math.floor(img.cols * crop);
        const y = Math.floor(img.rows * crop);
        const w = Math.floor(img.cols * (1 - 2 * crop));
        const h = Math.floor(img.rows * (1 - 2 * crop));
        processedImage = img.getRegion(new cv.Rect(x, y, w, h));
      }
      
      // Encode
      const buffer = cv.imencode('.jpg', processedImage);
      
      // Enhance with Sharp
      const enhanced = await sharp(buffer)
        .resize(2480, 3508, { fit: 'inside' })
        .normalize()
        .sharpen({ sigma: 1.0 })
        .toBuffer();
      
      // Add to PDF
      const pdfImage = await pdfDoc.embedJpg(enhanced);
      const page = pdfDoc.addPage([2480, 3508]);
      const { width: w, height: h } = pdfImage.scale(1);
      page.drawImage(pdfImage, {
        x: (2480 - w) / 2,
        y: (3508 - h) / 2,
        width: w,
        height: h
      });
    }
    
    const pdfBytes = await pdfDoc.save();
    
    res.setHeader('X-Debug-Info', JSON.stringify(debugInfo).substring(0, 8000));
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=scan.pdf');
    res.send(Buffer.from(pdfBytes));
    
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message, stack: error.stack });
  }
});

function orderPoints(pts) {
  // Sum of coordinates: TL has smallest, BR has largest
  const summed = pts.map(p => ({ p, sum: p[0] + p[1] }));
  summed.sort((a, b) => a.sum - b.sum);
  
  const tl = summed[0].p;
  const br = summed[3].p;
  
  // Difference: TR has positive diff, BL has negative
  const remaining = [summed[1].p, summed[2].p];
  const diffed = remaining.map(p => ({ p, diff: p[1] - p[0] }));
  diffed.sort((a, b) => a.diff - b.diff);
  
  const tr = diffed[1].p;
  const bl = diffed[0].p;
  
  return [tl, tr, br, bl];
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', opencv: cv.version });
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Scanner backend running on port ${PORT}`);
  console.log(`OpenCV version: ${cv.version.major}.${cv.version.minor}.${cv.version.revision}`);
});
