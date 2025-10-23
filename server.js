const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const { PDFDocument } = require('pdf-lib');
const cv = require('@u4/opencv4nodejs');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// Load corner detection model (you'll need to host this or use a pre-trained one)
let cornerModel = null;

async function loadModel() {
  try {
    // Option 1: Load from URL
    // cornerModel = await tf.loadGraphModel('https://your-model-url/model.json');
    
    // Option 2: Use simple heuristic model (fallback)
    console.log('Using heuristic corner detection (no ML model loaded)');
  } catch (e) {
    console.error('Failed to load ML model:', e);
  }
}

loadModel();

// ML-based corner detection
async function detectCornersML(imageBuffer) {
  if (!cornerModel) {
    return null; // Fall back to OpenCV
  }
  
  try {
    // Preprocess image for model
    const img = tf.node.decodeImage(imageBuffer);
    const resized = tf.image.resizeBilinear(img, [256, 256]);
    const normalized = resized.div(255.0).expandDims(0);
    
    // Run inference
    const prediction = await cornerModel.predict(normalized);
    const corners = await prediction.array();
    
    // corners format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    return corners[0];
  } catch (e) {
    console.error('ML detection failed:', e);
    return null;
  }
}

// Hough Transform for line detection (better than contours for documents)
function detectDocumentHough(img) {
  const gray = img.bgrToGray();
  const blurred = gray.gaussianBlur(new cv.Size(5, 5), 0);
  const edges = blurred.canny(50, 150);
  
  // Hough Line Transform - detects straight lines
  const lines = edges.houghLinesP(
    1,              // rho
    Math.PI / 180,  // theta
    100,            // threshold
    50,             // minLineLength
    10              // maxLineGap
  );
  
  if (!lines || lines.length < 4) {
    return null;
  }
  
  // Cluster lines into horizontal and vertical
  const horizontal = [];
  const vertical = [];
  
  for (const line of lines) {
    const angle = Math.atan2(line.w - line.y, line.z - line.x) * 180 / Math.PI;
    if (Math.abs(angle) < 10 || Math.abs(angle) > 170) {
      horizontal.push(line);
    } else if (Math.abs(Math.abs(angle) - 90) < 10) {
      vertical.push(line);
    }
  }
  
  if (horizontal.length < 2 || vertical.length < 2) {
    return null;
  }
  
  // Find extreme lines (top, bottom, left, right)
  const top = horizontal.sort((a, b) => a.y - b.y)[0];
  const bottom = horizontal.sort((a, b) => b.w - a.w)[0];
  const left = vertical.sort((a, b) => a.x - b.x)[0];
  const right = vertical.sort((a, b) => b.z - a.z)[0];
  
  // Calculate intersections to get corners
  const corners = [
    intersectLines(top, left),     // top-left
    intersectLines(top, right),    // top-right
    intersectLines(bottom, right), // bottom-right
    intersectLines(bottom, left)   // bottom-left
  ];
  
  return corners.filter(c => c !== null);
}

function intersectLines(line1, line2) {
  const x1 = line1.x, y1 = line1.y, x2 = line1.z, y2 = line1.w;
  const x3 = line2.x, y3 = line2.y, x4 = line2.z, y4 = line2.w;
  
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denom) < 0.001) return null;
  
  const x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom;
  const y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom;
  
  return [x, y];
}

app.post('/process-scan', upload.array('images'), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No images provided' });
    }

    const pdfDoc = await PDFDocument.create();
    const debugInfo = [];
    
    for (let file of req.files) {
      let img = cv.imdecode(file.buffer);
      const originalHeight = img.rows;
      const originalWidth = img.cols;
      
      debugInfo.push(`Processing ${originalWidth}x${originalHeight} image`);
      
      let corners = null;
      let method = 'none';
      
      // Try ML detection first
      corners = await detectCornersML(file.buffer);
      if (corners && corners.length === 4) {
        method = 'ML';
        debugInfo.push('✓ ML corner detection succeeded');
      } else {
        debugInfo.push('ML detection failed, trying Hough Transform');
        
        // Try Hough Transform
        corners = detectDocumentHough(img);
        if (corners && corners.length === 4) {
          method = 'Hough';
          debugInfo.push('✓ Hough Transform succeeded');
        } else {
          debugInfo.push('Hough failed, using fallback crop');
        }
      }
      
      let processedImage = img;
      
      if (corners && corners.length === 4) {
        // Order corners
        const ordered = orderPoints(corners);
        
        // Perspective transform
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
          debugInfo.push(`✓ Transformed using ${method}`);
        } catch (e) {
          debugInfo.push(`Transform failed: ${e.message}`);
          // Fallback crop
          const crop = 0.05;
          const x = Math.floor(img.cols * crop);
          const y = Math.floor(img.rows * crop);
          const w = Math.floor(img.cols * (1 - 2 * crop));
          const h = Math.floor(img.rows * (1 - 2 * crop));
          processedImage = img.getRegion(new cv.Rect(x, y, w, h));
        }
      } else {
        // Fallback: simple crop
        const crop = 0.05;
        const x = Math.floor(img.cols * crop);
        const y = Math.floor(img.rows * crop);
        const w = Math.floor(img.cols * (1 - 2 * crop));
        const h = Math.floor(img.rows * (1 - 2 * crop));
        processedImage = img.getRegion(new cv.Rect(x, y, w, h));
      }
      
      // Encode and enhance
      const buffer = cv.imencode('.jpg', processedImage);
      const enhanced = await sharp(buffer)
        .resize(2480, 3508, { fit: 'inside' })
        .normalize()
        .sharpen()
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
    
    res.setHeader('X-Debug-Info', JSON.stringify(debugInfo));
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=scan.pdf');
    res.send(Buffer.from(pdfBytes));
    
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message });
  }
});

function orderPoints(pts) {
  const summed = pts.map(p => ({ p, sum: p[0] + p[1] }));
  summed.sort((a, b) => a.sum - b.sum);
  
  const tl = summed[0].p;
  const br = summed[3].p;
  
  const remaining = [summed[1].p, summed[2].p];
  const diffed = remaining.map(p => ({ p, diff: p[1] - p[0] }));
  diffed.sort((a, b) => a.diff - b.diff);
  
  return [tl, diffed[1].p, br, diffed[0].p];
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', opencv: cv.version });
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Scanner backend running on port ${PORT}`);
});
