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
      
      debugInfo.push(`Original: ${originalWidth}x${originalHeight}`);
      
      // Resize for processing
      let workingImg = img;
      const maxDim = 1000;
      if (img.cols > maxDim || img.rows > maxDim) {
        const scale = maxDim / Math.max(img.cols, img.rows);
        workingImg = img.resize(
          Math.floor(img.rows * scale),
          Math.floor(img.cols * scale)
        );
      }
      
      // Convert to grayscale
      const gray = workingImg.bgrToGray();
      
      // Blur
      const blurred = gray.gaussianBlur(new cv.Size(5, 5), 0);
      
      // OTSU threshold (automatic)
      const thresh = blurred.threshold(0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
      
      // Check if document is light or dark - invert if needed
      const mean = thresh.mean();
      const binary = mean > 127 ? thresh.bitwiseNot() : thresh;
      
      // Morphology: remove noise, close gaps
      const kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
      const opened = binary.morphologyEx(kernel1, cv.MORPH_OPEN);
      
      const kernel2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(20, 20));
      const closed = opened.morphologyEx(kernel2, cv.MORPH_CLOSE);
      
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
      
      // Look for document (should be 30-95% of image)
      for (let i = 0; i < Math.min(10, sorted.length); i++) {
        const contour = sorted[i];
        const area = contour.area;
        const areaPercent = (area / imageArea) * 100;
        
        // Skip if too big (99%+ = whole image) or too small
        if (areaPercent > 98 || areaPercent < 25) {
          debugInfo.push(`Contour ${i}: ${areaPercent.toFixed(1)}% - skip (too ${areaPercent > 98 ? 'big' : 'small'})`);
          continue;
        }
        
        debugInfo.push(`Contour ${i}: ${areaPercent.toFixed(1)}% - checking...`);
        
        // Approximate
        const peri = contour.arcLength(true);
        
        // Try multiple epsilon values
        for (let eps = 0.01; eps <= 0.05; eps += 0.01) {
          const approx = contour.approxPolyDP(eps * peri, true);
          
          // Extract points
          let srcPoints = [];
          try {
            for (let j = 0; j < approx.rows; j++) {
              const pt = approx.at(j);
              srcPoints.push([pt.x, pt.y]);
            }
          } catch (e) {
            continue;
          }
          
          const numPoints = srcPoints.length;
          
          if (numPoints === 4) {
            debugInfo.push(`  ✓ Found 4 corners with ε=${eps.toFixed(2)}`);
            
            // Scale to original
            if (workingImg.cols !== originalWidth) {
              const scaleX = originalWidth / workingImg.cols;
              const scaleY = originalHeight / workingImg.rows;
              srcPoints = srcPoints.map(p => [p[0] * scaleX, p[1] * scaleY]);
            }
            
            // Order
            const ordered = orderPoints(srcPoints);
            
            // Transform
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
              debugInfo.push(`  ✗ Transform failed: ${e.message}`);
            }
          } else {
            debugInfo.push(`  ${numPoints} corners (need 4)`);
          }
        }
        
        if (transformed) break;
      }
      
      debugInfo.push(`Result: ${transformed ? 'TRANSFORMED ✓' : 'FALLBACK CROP'}`);
      
      // Fallback
      if (!transformed) {
        const crop = 0.02;
        const x = Math.floor(img.cols * crop);
        const y = Math.floor(img.rows * crop);
        const w = Math.floor(img.cols * (1 - 2 * crop));
        const h = Math.floor(img.rows * (1 - 2 * crop));
        processedImage = img.getRegion(new cv.Rect(x, y, w, h));
      }
      
      // Encode
      const buffer = cv.imencode('.jpg', processedImage);
      
      // Enhance
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
    
    res.setHeader('X-Debug-Info', JSON.stringify(debugInfo).substring(0, 8000));
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=scan.pdf');
    res.send(Buffer.from(pdfBytes));
    
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message });
  }
});
