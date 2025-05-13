/**
 * Improved texture mapping function for clothing visualization
 * This function properly maps textures onto 3D clothing models
 */
function applyTextureToClothingModel(model, textureUrl, options = {}) {
  // Default options
  const defaultOptions = {
    roughness: 0.4,            // Lower for fabric
    metalness: 0.0,            // Fabrics are not metallic
    normalMapStrength: 0.5,    // Subtle fabric texture
    normalMapUrl: null,        // Optional normal map
    colorIntensity: 1.0,       // Color brightness multiplier
    repeats: 1,                // Texture repeat count
    rotation: 0,               // Texture rotation in radians
    applyColor: true,          // Whether to apply dominant color
    dominantColor: null,       // Optional dominant color to tint texture
    debugMode: false           // Enable debug visualization
  };

  // Merge options
  const settings = Object.assign({}, defaultOptions, options);

  if (!model || !textureUrl) {
    console.error("Missing model or texture URL");
    return false;
  }

  // Create texture loader with better error handling
  const textureLoader = new THREE.TextureLoader();
  textureLoader.setCrossOrigin('anonymous');

  // Load texture with progress and error handling
  return new Promise((resolve, reject) => {
    // Show loading indicator if available
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) loadingOverlay.classList.add('visible');

    textureLoader.load(
      textureUrl,
      (texture) => {
        try {
          // Configure texture for better appearance
          texture.encoding = THREE.sRGBEncoding;
          texture.anisotropy = window.renderer ?
            window.renderer.capabilities.getMaxAnisotropy() : 16;
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
          texture.repeat.set(settings.repeats, settings.repeats);
          texture.rotation = settings.rotation;
          texture.needsUpdate = true;

          // Load normal map if provided
          let normalMapPromise = Promise.resolve(null);
          if (settings.normalMapUrl) {
            normalMapPromise = new Promise((resolveNormal, rejectNormal) => {
              textureLoader.load(
                settings.normalMapUrl,
                (normalMap) => {
                  normalMap.wrapS = THREE.RepeatWrapping;
                  normalMap.wrapT = THREE.RepeatWrapping;
                  normalMap.repeat.copy(texture.repeat);
                  normalMap.rotation = texture.rotation;
                  normalMap.needsUpdate = true;
                  resolveNormal(normalMap);
                },
                undefined,
                (error) => {
                  console.warn("Error loading normal map:", error);
                  resolveNormal(null); // Continue without normal map
                }
              );
            });
          }

          // Proceed when normal map is loaded (or not)
          normalMapPromise.then((normalMap) => {
            // Create material optimized for fabric visualization
            const material = new THREE.MeshStandardMaterial({
              map: texture,
              normalMap: normalMap,
              normalScale: new THREE.Vector2(settings.normalMapStrength, settings.normalMapStrength),
              roughness: settings.roughness,
              metalness: settings.metalness,
              side: THREE.DoubleSide, // Show both sides of fabric
              transparent: true
            });

            // Apply dominant color tinting if requested
            if (settings.applyColor && settings.dominantColor) {
              // Convert [B,G,R] to THREE.Color [R,G,B]
              if (Array.isArray(settings.dominantColor) && settings.dominantColor.length >= 3) {
                // Extract components (handles both [B,G,R] and [R,G,B] formats)
                let b, g, r;
                if (settings.dominantColor.length === 3) {
                  // Assume BGR order from OpenCV
                  [b, g, r] = settings.dominantColor;
                } else {
                  // Handle other formats
                  r = settings.dominantColor[0];
                  g = settings.dominantColor[1];
                  b = settings.dominantColor[2];
                }

                // Create THREE.Color (uses RGB order)
                const color = new THREE.Color(
                  r / 255.0,
                  g / 255.0,
                  b / 255.0
                );

                // Apply color intensity
                if (settings.colorIntensity !== 1.0) {
                  const hsl = {};
                  color.getHSL(hsl);
                  hsl.l = Math.min(1, hsl.l * settings.colorIntensity);
                  color.setHSL(hsl.h, hsl.s, hsl.l);
                }

                material.color = color;
              }
            }

            // Store original materials if not already stored
            model.traverse((node) => {
              if (node.isMesh) {
                if (!node.userData.originalMaterial) {
                  node.userData.originalMaterial = node.material;
                }

                // Apply new material
                node.material = material;

                // Enable shadows for better realism
                node.castShadow = true;
                node.receiveShadow = true;
              }
            });

            // Create debug visualization if requested
            if (settings.debugMode) {
              createMaterialDebugView(texture, normalMap, material, settings.dominantColor);
            }

            // Hide loading indicator
            if (loadingOverlay) loadingOverlay.classList.remove('visible');

            // Update UI to show texture is applied
            updateAppliedTextureUI(textureUrl);

            resolve(true);
          });
        } catch (error) {
          console.error("Error applying texture:", error);
          if (loadingOverlay) loadingOverlay.classList.remove('visible');
          reject(error);
        }
      },
      (progress) => {
        // Update progress if needed
        if (progress.lengthComputable) {
          const percent = Math.floor((progress.loaded / progress.total) * 100);
          updateLoadingProgress(percent);
        }
      },
      (error) => {
        console.error("Error loading texture:", error);
        if (loadingOverlay) loadingOverlay.classList.remove('visible');
        reject(error);
      }
    );
  });
}

/**
 * Helper function to create debug visualization for material
 */
function createMaterialDebugView(texture, normalMap, material, dominantColor) {
  // Find or create texture preview container
  let previewContainer = document.querySelector('.texture-preview-container');
  if (!previewContainer) {
    previewContainer = document.createElement('div');
    previewContainer.className = 'texture-preview-container';
    document.body.appendChild(previewContainer);
  }

  // Show the container
  previewContainer.style.display = 'block';

  // Clear previous content
  previewContainer.innerHTML = `
    <h3>Texture Preview</h3>
    <div class="texture-preview">
      <div class="texture-images">
        <div class="texture-image-container">
          <h4>Diffuse Texture</h4>
          <div class="texture-canvas-container">
            <canvas id="diffuse-preview" width="200" height="200"></canvas>
          </div>
        </div>
        ${normalMap ? `
        <div class="texture-image-container">
          <h4>Normal Map</h4>
          <div class="texture-canvas-container">
            <canvas id="normal-preview" width="200" height="200"></canvas>
          </div>
        </div>
        ` : ''}
      </div>
      <div class="texture-properties">
        <div class="color-preview" style="
          width: 100px;
          height: 100px;
          background-color: ${dominantColor ? `rgb(${dominantColor[2]}, ${dominantColor[1]}, ${dominantColor[0]})` : 'rgb(200,200,200)'};
          border-radius: 8px;
          border: 1px solid #ccc;
          display: inline-block;
          margin-right: 10px;
        "></div>
        <div style="display: inline-block; vertical-align: middle;">
          <p><strong>Material Settings:</strong></p>
          <p>Roughness: ${material.roughness.toFixed(2)}</p>
          <p>Metalness: ${material.metalness.toFixed(2)}</p>
          ${normalMap ? `<p>Normal Map Strength: ${material.normalScale.x.toFixed(2)}</p>` : ''}
        </div>
      </div>
    </div>
  `;

  // Draw texture to canvas
  setTimeout(() => {
    // Draw diffuse texture
    const diffuseCanvas = document.getElementById('diffuse-preview');
    if (diffuseCanvas && texture) {
      const ctx = diffuseCanvas.getContext('2d');
      drawTextureToCanvas(ctx, texture);
    }

    // Draw normal map
    const normalCanvas = document.getElementById('normal-preview');
    if (normalCanvas && normalMap) {
      const ctx = normalCanvas.getContext('2d');
      drawTextureToCanvas(ctx, normalMap);
    }
  }, 100);
}

/**
 * Helper function to draw Three.js texture to canvas
 */
function drawTextureToCanvas(ctx, texture) {
  if (!ctx || !texture || !texture.image) return;

  // Clear canvas
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw texture
  const img = texture.image;
  const scale = Math.min(
    ctx.canvas.width / img.width,
    ctx.canvas.height / img.height
  );

  const width = img.width * scale;
  const height = img.height * scale;
  const x = (ctx.canvas.width - width) / 2;
  const y = (ctx.canvas.height - height) / 2;

  ctx.drawImage(img, x, y, width, height);
}

/**
 * Update UI to show applied texture
 */
function updateAppliedTextureUI(textureUrl) {
  // Find or create status element
  let statusElement = document.getElementById('texture-status');
  if (!statusElement) {
    statusElement = document.createElement('div');
    statusElement.id = 'texture-status';
    statusElement.className = 'alert alert-success mt-3';

    // Find a good place to insert it
    const container = document.querySelector('.viewer-controls') ||
                     document.getElementById('canvas-container');
    if (container) {
      container.appendChild(statusElement);
    }
  }

  // Update status
  statusElement.innerHTML = `
    <strong>Success!</strong> Texture applied from: 
    <span class="texture-path">${textureUrl.split('/').pop()}</span>
    <button class="btn btn-sm btn-outline-primary ml-2" 
            onclick="document.querySelector('.texture-preview-container').style.display = 'block'">
      Show Texture Details
    </button>
  `;
  statusElement.style.display = 'block';
}

/**
 * Update loading progress
 */
function updateLoadingProgress(percent) {
  const loadingText = document.getElementById('loading-text');
  if (loadingText) {
    loadingText.textContent = `Loading... ${percent}%`;
  }
}

/**
 * Process real clothing image with enhanced color detection
 */
async function processRealClothingEnhanced() {
  // Get file input and clothing type
  const fileInput = document.getElementById('real-clothing-upload');
  const clothingType = document.getElementById('clothing-type-select').value;

  // Check if file selected
  if (!fileInput || !fileInput.files[0]) {
    showError("Please select an image first");
    return;
  }

  // Show detailed loading indicator
  showLoading("Processing clothing texture...");

  try {
    // Create form data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('clothing_type', clothingType);

    // Send to API
    const response = await fetch('/api/process-clothing-texture', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || "Unknown error");
    }

    // Get the appropriate model path
    const modelPath = data.model_path || getModelPathForType(clothingType);

    // Load the model
    const model = await loadModel(modelPath);

    // Apply texture with enhanced function
    await applyTextureToClothingModel(model, data.texture_url, {
      normalMapUrl: data.normal_map_url,
      roughness: 0.4,  // Good value for fabric
      metalness: 0.0,  // Fabric isn't metallic
      normalMapStrength: 0.8,
      debugMode: true  // Show debug visualization
    });

    // Provide success feedback
    showSuccessMessage("Texture applied successfully!");

  } catch (error) {
    console.error("Error processing clothing:", error);
    showError(`Error processing clothing: ${error.message}`);
  } finally {
    hideLoading();
  }
}

/**
 * Show success message
 */
function showSuccessMessage(message) {
  // Create success toast if not exists
  let successToast = document.getElementById('success-toast');

  if (!successToast) {
    successToast = document.createElement('div');
    successToast.id = 'success-toast';
    successToast.style.position = 'fixed';
    successToast.style.bottom = '20px';
    successToast.style.right = '20px';
    successToast.style.backgroundColor = '#4CAF50';
    successToast.style.color = 'white';
    successToast.style.padding = '15px';
    successToast.style.borderRadius = '4px';
    successToast.style.zIndex = '1000';
    successToast.style.transition = 'opacity 0.5s';
    successToast.style.opacity = '0';

    document.body.appendChild(successToast);
  }

  // Set message
  successToast.textContent = message;

  // Show toast
  successToast.style.opacity = '1';

  // Hide after 5 seconds
  setTimeout(() => {
    successToast.style.opacity = '0';
  }, 5000);
}

// Replace the Apply button click handler
document.addEventListener('DOMContentLoaded', function() {
  const applyButton = document.getElementById('apply-real-clothing-btn');
  if (applyButton) {
    // Replace existing click listener with our enhanced version
    applyButton.onclick = null;
    applyButton.addEventListener('click', processRealClothingEnhanced);
  }
});