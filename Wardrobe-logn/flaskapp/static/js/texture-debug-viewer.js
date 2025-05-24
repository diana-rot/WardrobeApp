/**
 * Texture debug viewer to visualize how textures are applied to models
 * This helps diagnose texture mapping issues
 */
class TextureDebugViewer {
  constructor(containerSelector = '#texture-debug-container') {
    // Create container if not exists
    this.container = document.querySelector(containerSelector);
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = containerSelector.replace('#', '');
      this.container.className = 'texture-debug-container';
      document.body.appendChild(this.container);

      // Style the container
      this.container.style.position = 'fixed';
      this.container.style.bottom = '20px';
      this.container.style.left = '20px';
      this.container.style.width = '400px';
      this.container.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
      this.container.style.border = '1px solid #ccc';
      this.container.style.borderRadius = '5px';
      this.container.style.padding = '10px';
      this.container.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
      this.container.style.zIndex = '1000';
      this.container.style.display = 'none'; // Initially hidden
    }

    // Initialize UI
    this.initUI();

    // Track textures
    this.textures = [];
  }

  /**
   * Initialize the user interface
   */
  initUI() {
    this.container.innerHTML = `
      <div class="texture-debug-header">
        <h3 style="margin: 0 0 10px 0; display: flex; justify-content: space-between; align-items: center;">
          <span>Texture Debug View</span>
          <button id="texture-debug-close" style="background: none; border: none; font-size: 20px; cursor: pointer;">×</button>
        </h3>
        <p style="margin: 0 0 10px 0; color: #666; font-size: 12px;">
          Visualize how textures are applied to your 3D model
        </p>
      </div>
      <div class="texture-debug-content">
        <div class="texture-preview-tabs" style="display: flex; border-bottom: 1px solid #ddd; margin-bottom: 10px;">
          <button class="preview-tab active" data-tab="original" style="flex: 1; padding: 5px; border: none; background: none; cursor: pointer; border-bottom: 2px solid #333;">
            Original
          </button>
          <button class="preview-tab" data-tab="applied" style="flex: 1; padding: 5px; border: none; background: none; cursor: pointer;">
            Applied
          </button>
          <button class="preview-tab" data-tab="uv" style="flex: 1; padding: 5px; border: none; background: none; cursor: pointer;">
            UV Map
          </button>
        </div>
        <div class="texture-preview-panels">
          <div class="preview-panel active" data-panel="original">
            <canvas id="original-texture-canvas" width="380" height="380" style="border: 1px solid #ddd; background-color: #f5f5f5;"></canvas>
          </div>
          <div class="preview-panel" data-panel="applied" style="display: none;">
            <canvas id="applied-texture-canvas" width="380" height="380" style="border: 1px solid #ddd; background-color: #f5f5f5;"></canvas>
          </div>
          <div class="preview-panel" data-panel="uv" style="display: none;">
            <canvas id="uv-map-canvas" width="380" height="380" style="border: 1px solid #ddd; background-color: #f5f5f5;"></canvas>
          </div>
        </div>
        <div class="texture-debug-info" style="margin-top: 10px;">
          <h4 style="margin: 5px 0;">Material Properties</h4>
          <div class="material-props" style="font-size: 12px; color: #333;">
            <div style="display: flex; justify-content: space-between;">
              <span>Color:</span> <span id="material-color">N/A</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span>Roughness:</span> <span id="material-roughness">N/A</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span>Metalness:</span> <span id="material-metalness">N/A</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span>Normal Map:</span> <span id="material-normal">N/A</span>
            </div>
          </div>
        </div>
      </div>
      <div class="texture-debug-actions" style="margin-top: 15px; display: flex; justify-content: space-between;">
        <button id="export-texture-btn" style="padding: 5px 10px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">
          Export Texture
        </button>
        <button id="reload-texture-btn" style="padding: 5px 10px; background-color: #2196F3; color: white; border: none; border-radius: 3px; cursor: pointer;">
          Reload
        </button>
      </div>
    `;

    // Add event listeners
    this.container.querySelector('#texture-debug-close').addEventListener('click', () => {
      this.hide();
    });

    // Tab switching
    this.container.querySelectorAll('.preview-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        // Update tab styling
        this.container.querySelectorAll('.preview-tab').forEach(t => {
          t.classList.remove('active');
          t.style.borderBottom = 'none';
        });
        tab.classList.add('active');
        tab.style.borderBottom = '2px solid #333';

        // Show corresponding panel
        const tabName = tab.getAttribute('data-tab');
        this.container.querySelectorAll('.preview-panel').forEach(panel => {
          if (panel.getAttribute('data-panel') === tabName) {
            panel.style.display = 'block';
          } else {
            panel.style.display = 'none';
          }
        });
      });
    });

    // Export button
    this.container.querySelector('#export-texture-btn').addEventListener('click', () => {
      this.exportTexture();
    });

    // Reload button
    this.container.querySelector('#reload-texture-btn').addEventListener('click', () => {
      this.refreshView();
    });
  }

  /**
   * Show the debug viewer
   */
  show() {
    this.container.style.display = 'block';
    this.refreshView();
  }

  /**
   * Hide the debug viewer
   */
  hide() {
    this.container.style.display = 'none';
  }

  /**
   * Add a texture to debug
   * @param {Object} textureInfo - Information about the texture
   */
  addTexture(textureInfo) {
    this.textures.push(textureInfo);
    this.refreshView();
  }

  /**
   * Refresh the debug view
   */
  refreshView() {
    if (this.textures.length === 0) return;

    // Get latest texture info
    const textureInfo = this.textures[this.textures.length - 1];

    // Update original texture canvas
    this.updateOriginalTextureCanvas(textureInfo.texture);

    // Update applied texture canvas
    this.updateAppliedTextureCanvas(textureInfo.material);

    // Update UV map canvas
    this.updateUVMapCanvas(textureInfo.model);

    // Update material properties
    this.updateMaterialProperties(textureInfo.material);
  }

  /**
   * Update original texture canvas
   * @param {THREE.Texture} texture - Original texture
   */
  updateOriginalTextureCanvas(texture) {
    const canvas = this.container.querySelector('#original-texture-canvas');
    if (!canvas || !texture || !texture.image) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw texture
    const img = texture.image;
    const scale = Math.min(
      canvas.width / img.width,
      canvas.height / img.height
    );

    const width = img.width * scale;
    const height = img.height * scale;
    const x = (canvas.width - width) / 2;
    const y = (canvas.height - height) / 2;

    ctx.drawImage(img, x, y, width, height);

    // Add border
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);

    // Add texture info
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(x, y + height - 30, width, 30);
    ctx.fillStyle = 'white';
    ctx.font = '12px Arial';
    ctx.fillText(`Original: ${img.width} × ${img.height}`, x + 5, y + height - 10);
  }

  /**
   * Update applied texture canvas
   * @param {THREE.Material} material - Material with texture applied
   */
  updateAppliedTextureCanvas(material) {
    const canvas = this.container.querySelector('#applied-texture-canvas');
    if (!canvas || !material || !material.map) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Create a tiny scene to render the material
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(canvas.width, canvas.height);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f5);

    // Create camera
    const camera = new THREE.PerspectiveCamera(45, canvas.width / canvas.height, 0.1, 100);
    camera.position.z = 5;

    // Create lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Create mesh with the material
    const geometry = new THREE.SphereGeometry(2, 32, 32);
    const mesh = new THREE.Mesh(geometry, material.clone());
    scene.add(mesh);

    // Render the scene
    renderer.render(scene, camera);

    // Draw the rendered image to our canvas
    ctx.drawImage(renderer.domElement, 0, 0);

    // Clean up
    renderer.dispose();
  }

  /**
   * Update UV map canvas
   * @param {THREE.Object3D} model - Model with UV mapping
   */
  updateUVMapCanvas(model) {
    const canvas = this.container.querySelector('#uv-map-canvas');
    if (!canvas || !model) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Find meshes in the model
    let uvData = [];
    model.traverse(node => {
      if (node.isMesh && node.geometry) {
        const geo = node.geometry;
        if (geo.attributes.uv) {
          // Extract UV coordinates
          const uvs = geo.attributes.uv.array;
          const indices = geo.index ? geo.index.array : null;

          // Format: [u1, v1, u2, v2, u3, v3, ...]
          for (let i = 0; i < uvs.length; i += 2) {
            uvData.push({
              u: uvs[i],
              v: 1 - uvs[i + 1] // Flip V since THREE.js uses bottom-left as origin
            });
          }
        }
      }
    });

    // Draw UV map
    if (uvData.length > 0) {
      ctx.fillStyle = 'rgba(200, 200, 200, 0.5)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw grid
      ctx.strokeStyle = 'rgba(150, 150, 150, 0.5)';
      ctx.lineWidth = 1;

      // Draw grid lines
      for (let i = 0; i <= 10; i++) {
        const pos = i / 10 * canvas.width;

        // Vertical line
        ctx.beginPath();
        ctx.moveTo(pos, 0);
        ctx.lineTo(pos, canvas.height);
        ctx.stroke();

        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(0, pos);
        ctx.lineTo(canvas.width, pos);
        ctx.stroke();
      }

      // Draw UV points
      ctx.fillStyle = 'rgba(0, 100, 255, 0.7)';
      for (const uv of uvData) {
        const x = uv.u * canvas.width;
        const y = uv.v * canvas.height;

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Add info text
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(5, canvas.height - 30, 150, 25);
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(`${uvData.length / 2} UV coordinates`, 10, canvas.height - 15);
    } else {
      // No UV data
      ctx.fillStyle = '#ddd';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = '#666';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('No UV mapping data available', canvas.width / 2, canvas.height / 2);
    }
  }

  /**
   * Update material properties display
   * @param {THREE.Material} material - Material to display
   */
  updateMaterialProperties(material) {
    if (!material) return;

    // Update color
    const colorElem = this.container.querySelector('#material-color');
    if (colorElem && material.color) {
      const color = material.color;
      const r = Math.round(color.r * 255);
      const g = Math.round(color.g * 255);
      const b = Math.round(color.b * 255);

      colorElem.innerHTML = `
        <span style="display: inline-block; width: 15px; height: 15px; background-color: rgb(${r}, ${g}, ${b}); border: 1px solid #ccc; vertical-align: middle;"></span>
        RGB(${r}, ${g}, ${b})
      `;
    }

    // Update roughness
    const roughnessElem = this.container.querySelector('#material-roughness');
    if (roughnessElem) {
      roughnessElem.textContent = material.roughness !== undefined ?
        material.roughness.toFixed(2) : 'N/A';
    }

    // Update metalness
    const metalnessElem = this.container.querySelector('#material-metalness');
    if (metalnessElem) {
      metalnessElem.textContent = material.metalness !== undefined ?
        material.metalness.toFixed(2) : 'N/A';
    }

    // Update normal map
    const normalElem = this.container.querySelector('#material-normal');
    if (normalElem) {
      normalElem.textContent = material.normalMap ?
        `Enabled (${material.normalScale.x.toFixed(2)})` : 'None';
    }
  }

  /**
   * Export texture to file
   */
  exportTexture() {
    if (this.textures.length === 0) return;

    const textureInfo = this.textures[this.textures.length - 1];
    if (!textureInfo.texture || !textureInfo.texture.image) return;

    // Create a canvas to draw the texture
    const canvas = document.createElement('canvas');
    const img = textureInfo.texture.image;
    canvas.width = img.width;
    canvas.height = img.height;

    // Draw the image to canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Create download link
    const link = document.createElement('a');
    link.download = 'exported_texture.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  }
}

// Create a global instance when the page loads
let textureDebugViewer;

document.addEventListener('DOMContentLoaded', () => {
  textureDebugViewer = new TextureDebugViewer();

  // Add buttons to toggle the viewer
  const viewerControls = document.querySelector('.viewer-controls');
  if (viewerControls) {
    const debugButton = document.createElement('button');
    debugButton.className = 'btn btn-info mt-3';
    debugButton.textContent = 'Debug Texture Mapping';
    debugButton.onclick = () => textureDebugViewer.show();
    viewerControls.appendChild(debugButton);
  }
});

// Function to capture texture data for debugging
function captureTextureForDebug(texture, material, model) {
  if (!textureDebugViewer) return;

  textureDebugViewer.addTexture({
    texture: texture,
    material: material,
    model: model
  });

  // Show the debug viewer
  textureDebugViewer.show();
}

// Modify the applyTextureToClothingModel function to capture texture data
const originalApplyTexture = window.applyTextureToClothingModel;
window.applyTextureToClothingModel = async function(model, textureUrl, options = {}) {
  const result = await originalApplyTexture(model, textureUrl, options);

  // Find the texture and material on the model
  let texture, material;
  model.traverse(node => {
    if (node.isMesh && node.material) {
      material = node.material;
      texture = node.material.map;
    }
  });

  // Capture for debugging
  if (texture && material) {
    captureTextureForDebug(texture, material, model);
  }

  return result;
};