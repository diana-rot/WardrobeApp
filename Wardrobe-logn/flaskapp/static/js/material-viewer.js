// Global variables for scene, camera, renderer, etc.
let scene, camera, renderer, controls, currentModel;
let textureManager;

// Initialize application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);

function init() {
  // Initialize TextureManager
  textureManager = new TextureManager();

  // Setup Three.js scene
  setupScene();

  // Setup event listeners
  setupEventListeners();

  // Load initial item if selected
  const itemSelect = document.getElementById('item-select');
  if (itemSelect && itemSelect.value) {
    loadItem(itemSelect.value);
  }
}

/**
 * Setup Three.js scene
 */
function setupScene() {
  const container = document.getElementById('canvas-container');
  if (!container) return;

  // Create scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f5f5);

  // Create camera
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 0, 3);

  // Create renderer
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance'
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap at 2x for performance
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // Make renderer globally available for TextureManager
  window.renderer = renderer;

  // Add renderer to container
  container.appendChild(renderer.domElement);

  // Create controls using the imported OrbitControls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Add lights
  addLights();

  // Start animation loop
  animate();

  // Handle window resize
  window.addEventListener('resize', onWindowResize);
}

/**
 * Add lights to scene
 */
function addLights() {
  // Add ambient light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  // Add key light
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
  keyLight.position.set(1, 2, 3);
  keyLight.castShadow = true;
  scene.add(keyLight);

  // Add fill light
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
  fillLight.position.set(-1, 1, -1);
  scene.add(fillLight);

  // Add rim light for better material definition
  const rimLight = new THREE.DirectionalLight(0xffffff, 0.2);
  rimLight.position.set(0, -1, -2);
  scene.add(rimLight);
}

/**
 * Animation loop
 */
function animate() {
  requestAnimationFrame(animate);

  // Update controls
  if (controls) {
    controls.update();
  }

  // Render scene
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

/**
 * Handle window resize
 */
function onWindowResize() {
  // Get container dimensions
  const container = document.getElementById('canvas-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Update camera
  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  // Update renderer
  renderer.setSize(width, height);
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Item selection
  const itemSelect = document.getElementById('item-select');
  if (itemSelect) {
    itemSelect.addEventListener('change', (event) => {
      loadItem(event.target.value);
    });
  }

  // Model selection
  const modelSelect = document.getElementById('model-select');
  if (modelSelect) {
    modelSelect.addEventListener('change', (event) => {
      loadModel(getModelPathForType(event.target.value));
    });
  }

  // Update material button
  const updateBtn = document.getElementById('update-material-btn');
  if (updateBtn) {
    updateBtn.addEventListener('click', updateMaterialFromUI);
  }

  // Real clothing application
  const applyRealClothingBtn = document.getElementById('apply-real-clothing-btn');
  if (applyRealClothingBtn) {
    applyRealClothingBtn.addEventListener('click', processRealClothing);
  }

  // Sliders for material properties
  setupSliderListeners();
}

/**
 * Setup listeners for material property sliders
 */
function setupSliderListeners() {
  const sliders = [
    'roughness-slider',
    'metalness-slider',
    'color-intensity-slider',
    'pattern-scale-slider',
    'pattern-rotation-slider',
    'normal-strength-slider'
  ];

  sliders.forEach(sliderId => {
    const slider = document.getElementById(sliderId);
    const valueId = sliderId.replace('-slider', '-value');

    if (slider) {
      slider.addEventListener('input', () => {
        updateSliderValue(sliderId, valueId);
      });
    }
  });
}

/**
 * Update slider value display
 * @param {string} sliderId - Slider element ID
 * @param {string} valueId - Value display element ID
 */
function updateSliderValue(sliderId, valueId) {
  const slider = document.getElementById(sliderId);
  const valueElement = document.getElementById(valueId);

  if (slider && valueElement) {
    // Special case for rotation (show in degrees)
    if (sliderId === 'pattern-rotation-slider') {
      valueElement.textContent = `${slider.value}°`;
    } else {
      valueElement.textContent = parseFloat(slider.value).toFixed(1);
    }
  }
}

/**
 * Load item by ID
 * @param {string} itemId - Item ID
 */
async function loadItem(itemId) {
  if (!itemId) return;

  try {
    // Show loading indicator
    showLoading("Loading item...");

    // Fetch item data
    const data = await fetchItemData(itemId);

    // Clear current model
    clearCurrentModel();

    // Get model path
    const modelPath = data.texture_data?.model_path ||
                     getModelPathForType(data.label);

    // Load model
    const model = await loadModel(modelPath, false); // Don't add to scene yet

    // Load material
    const material = await loadMaterial(data);

    // Apply material to model
    if (material) {
      textureManager.applyMaterial(model, material);
    }

    // Add model to scene
    scene.add(model);
    currentModel = model;

    // Update material controls
    updateMaterialControls(data.materialProperties);

    // Hide loading indicator
    hideLoading();

  } catch (error) {
    console.error("Error loading item:", error);
    hideLoading();
    showError(`Failed to load item: ${error.message}`);
  }
}

/**
 * Fetch item data from API
 * @param {string} itemId - Item ID
 * @returns {Promise<Object>} - Item data
 */
async function fetchItemData(itemId) {
  const response = await fetch('/api/wardrobe/process-clothing', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      itemId: itemId,
      imageUrl: `/static/image_users/${itemId}/original.jpg`, // Assuming this is the path structure
      itemType: 'T-shirt/top' // Default to t-shirt, adjust based on your needs
    })
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch item data: ${response.statusText}`);
  }

  const data = await response.json();

  if (!data.success) {
    throw new Error(data.error || "Unknown error");
  }

  return data.modelData; // Return the modelData object which contains the texture and model information
}

/**
 * Load model
 * @param {string} modelPath - Model path
 * @param {boolean} addToScene - Whether to add to scene
 * @returns {Promise<THREE.Object3D>} - Loaded model
 */
async function loadModel(modelPath, addToScene = true) {
  // Show loading indicator
  showLoading("Loading model...");

  try {
    // Check for primitive shapes
    if (['sphere', 'cube', 'torus', 'cylinder'].includes(modelPath)) {
      const model = createPrimitive(modelPath);

      if (addToScene) {
        // Clear current model
        clearCurrentModel();

        // Add to scene
        scene.add(model);
        currentModel = model;
      }

      hideLoading();
      return model;
    }

    // Load GLTF model
    return new Promise((resolve, reject) => {
      const loader = new THREE.GLTFLoader();

      loader.load(
        modelPath,
        (gltf) => {
          const model = gltf.scene;

          // Optimize model
          optimizeModel(model);

          if (addToScene) {
            // Clear current model
            clearCurrentModel();

            // Add to scene
            scene.add(model);
            currentModel = model;
          }

          hideLoading();
          resolve(model);
        },
        (xhr) => {
          // Update loading progress
          const percent = xhr.lengthComputable ?
            Math.floor((xhr.loaded / xhr.total) * 100) : 0;
          updateLoadingProgress(percent);
        },
        (error) => {
          console.error("Error loading model:", error);
          hideLoading();
          reject(error);
        }
      );
    });
  } catch (error) {
    console.error("Error loading model:", error);
    hideLoading();
    throw error;
  }
}

/**
 * Create primitive shape
 * @param {string} type - Primitive type
 * @returns {THREE.Object3D} - Primitive object
 */
function createPrimitive(type) {
  let geometry;

  switch (type) {
    case 'sphere':
      geometry = new THREE.SphereGeometry(1, 32, 32);
      break;
    case 'cube':
      geometry = new THREE.BoxGeometry(1, 1, 1);
      break;
    case 'torus':
      geometry = new THREE.TorusGeometry(0.8, 0.2, 16, 48);
      break;
    case 'cylinder':
      geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 32);
      break;
    default:
      geometry = new THREE.SphereGeometry(1, 32, 32);
  }

  // Create basic material
  const material = new THREE.MeshStandardMaterial({
    color: 0xcccccc,
    roughness: 0.7,
    metalness: 0.0
  });

  // Create mesh
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return mesh;
}

/**
 * Optimize model for better performance
 * @param {THREE.Object3D} model - Model to optimize
 */
function optimizeModel(model) {
  if (!model) return;

  // Center model
  const box = new THREE.Box3().setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());

  // Calculate appropriate scale
  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = 2 / maxDim;

  // Apply scale
  model.scale.set(scale, scale, scale);

  // Center model
  model.position.set(
    -center.x * scale,
    -center.y * scale,
    -center.z * scale
  );

  // Optimize meshes
  model.traverse((node) => {
    if (node.isMesh) {
      // Enable shadows
      node.castShadow = true;
      node.receiveShadow = true;

      // Optimize geometry
      if (node.geometry) {
        // Disable unnecessary updates
        node.geometry.attributes.position.needsUpdate = false;
        if (node.geometry.attributes.normal) {
          node.geometry.attributes.normal.needsUpdate = false;
        }
        if (node.geometry.attributes.uv) {
          node.geometry.attributes.uv.needsUpdate = false;
        }
      }
    }
  });
}

/**
 * Load material for item
 * @param {Object} data - Item data
 * @returns {Promise<THREE.Material>} - Material
 */
async function loadMaterial(data) {
  try {
    // Create texture data object
    const textureData = {
      texture_path: data.textureUrl,
      normal_map_path: data.normalMapUrl
    };

    // Use TextureManager to create material
    return await textureManager.createMaterial(
      textureData,
      {
        roughness: 0.7,
        metalness: 0.0,
        normalStrength: 1.0,
        colorIntensity: 1.0,
        patternScale: 1.0,
        patternRotation: 0
      }
    );
  } catch (error) {
    console.error("Error loading material:", error);
    throw error;
  }
}

/**
 * Update material controls based on properties
 * @param {Object} materialProps - Material properties
 */
function updateMaterialControls(materialProps) {
  if (!materialProps) return;

  // Get properties
  const {
    estimated_material,
    texture_variance = 0.5,
    edge_density = 0.1,
    pattern_info = {}
  } = materialProps;

  // Get default values based on material type
  const defaults = textureManager.getMaterialSettings(estimated_material);

  // Update sliders
  updateSlider('roughness-slider', 'roughness-value', defaults.roughness);
  updateSlider('metalness-slider', 'metalness-value', defaults.metalness);
  updateSlider('normal-strength-slider', 'normal-strength-value', defaults.normalStrength);
  updateSlider('color-intensity-slider', 'color-intensity-value', 1.0);

  // Update pattern sliders based on pattern_info
  if (pattern_info.has_pattern) {
    // Pattern scale based on pattern_scale
    let scaleValue = 1.0;
    if (pattern_info.pattern_scale === 'fine') {
      scaleValue = 0.5;
    } else if (pattern_info.pattern_scale === 'large') {
      scaleValue = 2.0;
    }

    updateSlider('pattern-scale-slider', 'pattern-scale-value', scaleValue);
    updateSlider('pattern-rotation-slider', 'pattern-rotation-value', 0);
  }

  // Update checkboxes
  document.getElementById('use-color-checkbox').checked = true;
  document.getElementById('use-normal-map-checkbox').checked = true;
}

/**
 * Update material from UI controls
 */
function updateMaterialFromUI() {
  // Get current model
  if (!currentModel) {
    showError("No model loaded");
    return;
  }

  // Get settings from UI
  const settings = {
    roughness: parseFloat(document.getElementById('roughness-slider').value),
    metalness: parseFloat(document.getElementById('metalness-slider').value),
    colorIntensity: parseFloat(document.getElementById('color-intensity-slider').value),
    patternScale: parseFloat(document.getElementById('pattern-scale-slider').value),
    patternRotation: parseFloat(document.getElementById('pattern-rotation-slider').value) * (Math.PI / 180),
    normalStrength: parseFloat(document.getElementById('normal-strength-slider').value),
    useColor: document.getElementById('use-color-checkbox').checked,
    useNormalMap: document.getElementById('use-normal-map-checkbox').checked
  };

  // Use TextureManager to update material
  textureManager.updateMaterialSettings(currentModel, settings);
}

/**
 * Update slider and value display
 * @param {string} sliderId - Slider element ID
 * @param {string} valueId - Value display element ID
 * @param {number} value - Value to set
 */
function updateSlider(sliderId, valueId, value) {
  const slider = document.getElementById(sliderId);
  const valueElement = document.getElementById(valueId);

  if (slider) {
    slider.value = value;
  }
if (valueElement) {
    if (sliderId === 'pattern-rotation-slider') {
      valueElement.textContent = `${Math.round(value)}°`;
    } else {
      valueElement.textContent = typeof value === 'number' ?
        value.toFixed(1) : value.toString();
    }
  }
}

/**
 * Process real clothing image
 */
async function processRealClothing() {
  // Get file input and clothing type
  const fileInput = document.getElementById('real-clothing-upload');
  const clothingType = document.getElementById('clothing-type-select').value;

  // Check if file selected
  if (!fileInput || !fileInput.files[0]) {
    showError("Please select an image first");
    return;
  }

  // Show processing status
  const statusElement = document.getElementById('texture-processing-status');
  if (statusElement) {
    statusElement.style.display = 'block';
  }

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

    // Create texture data object
    const textureData = {
      texture_path: data.texture_url,
      normal_map_path: data.normal_map_url
    };

    // Create default material properties
    const materialProps = createDefaultMaterialProps(clothingType);

    // Use TextureManager to create material
    const material = await textureManager.createMaterial(textureData, materialProps);

    // Load appropriate model
    const modelPath = data.model_path || getModelPathForType(clothingType);
    const model = await loadModel(modelPath, false); // Don't add to scene yet

    // Apply material
    textureManager.applyMaterial(model, material);

    // Clear current model
    clearCurrentModel();

    // Add to scene
    scene.add(model);
    currentModel = model;

    // Update controls with default values
    updateMaterialControls(materialProps);

  } catch (error) {
    console.error("Error processing clothing:", error);
    showError(`Error processing clothing: ${error.message}`);
  } finally {
    // Hide processing status
    if (statusElement) {
      statusElement.style.display = 'none';
    }
  }
}

/**
 * Create default material properties based on clothing type
 * @param {string} clothingType - Type of clothing
 * @returns {Object} - Default material properties
 */
function createDefaultMaterialProps(clothingType) {
  // Default properties by clothing type
  const defaults = {
    'T-shirt/top': {
      estimated_material: 'medium',
      texture_variance: 100,
      edge_density: 0.1,
      primary_color_rgb: [200, 200, 200],
      pattern_info: {
        pattern_type: "regular",
        pattern_scale: "medium",
        pattern_strength: 0.3,
        has_pattern: false,
        pattern_regularity: 0.5,
        is_directional: false,
        peak_count: 0
      }
    },
    'Trouser': {
      estimated_material: 'textured',
      texture_variance: 150,
      edge_density: 0.2,
      primary_color_rgb: [100, 100, 150],
      pattern_info: {
        pattern_type: "regular",
        pattern_scale: "medium",
        pattern_strength: 0.2,
        has_pattern: false,
        pattern_regularity: 0.5,
        is_directional: false,
        peak_count: 0
      }
    },
    'Pullover': {
      estimated_material: 'textured',
      texture_variance: 120,
      edge_density: 0.15,
      primary_color_rgb: [150, 150, 150],
      pattern_info: {
        pattern_type: "regular",
        pattern_scale: "medium",
        pattern_strength: 0.4,
        has_pattern: false,
        pattern_regularity: 0.5,
        is_directional: false,
        peak_count: 0
      }
    },
    'Dress': {
      estimated_material: 'smooth',
      texture_variance: 80,
      edge_density: 0.08,
      primary_color_rgb: [200, 150, 150],
      pattern_info: {
        pattern_type: "regular",
        pattern_scale: "medium",
        pattern_strength: 0.3,
        has_pattern: false,
        pattern_regularity: 0.5,
        is_directional: false,
        peak_count: 0
      }
    },
    'Coat': {
      estimated_material: 'textured',
      texture_variance: 180,
      edge_density: 0.25,
      primary_color_rgb: [100, 100, 100],
      pattern_info: {
        pattern_type: "regular",
        pattern_scale: "medium",
        pattern_strength: 0.2,
        has_pattern: false,
        pattern_regularity: 0.5,
        is_directional: false,
        peak_count: 0
      }
    }
  };

  // Return defaults for clothing type or use T-shirt as fallback
  return defaults[clothingType] || defaults['T-shirt/top'];
}

/**
 * Get model path for clothing type
 * @param {string} clothingType - Clothing type
 * @returns {string} - Model path
 */
function getModelPathForType(clothingType) {
  // Map clothing types to model paths
  const modelMap = {
    'T-shirt/top': '/static/models/clothing/tshirt.glb',
    'Trouser': '/static/models/clothing/trouser.glb',
    'Pullover': '/static/models/clothing/pullover.glb',
    'Dress': '/static/models/clothing/dress.glb',
    'Coat': '/static/models/clothing/coat.glb',
    'Sandal': '/static/models/clothing/sandal.glb',
    'Shirt': '/static/models/clothing/shirt.glb',
    'Sneaker': '/static/models/clothing/sneaker.glb',
    'Bag': '/static/models/clothing/bag.glb',
    'Ankle boot': '/static/models/clothing/boot.glb',

    // Map primitive shape names directly
    'sphere': 'sphere',
    'cube': 'cube',
    'torus': 'torus',
    'cylinder': 'cylinder'
  };

  // Return model path or default to T-shirt
  return modelMap[clothingType] || '/static/models/clothing/tshirt.glb';
}

/**
 * Clear current model
 */
function clearCurrentModel() {
  if (currentModel) {
    scene.remove(currentModel);

    // Allow TextureManager to clean up
    textureManager.restoreOriginalMaterials(currentModel);

    currentModel = null;
  }
}

/**
 * Show loading indicator
 * @param {string} message - Optional message
 */
function showLoading(message) {
  const loading = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');

  if (loading) {
    loading.style.display = 'flex';
  }

  if (loadingText && message) {
    loadingText.textContent = message;
  }
}

/**
 * Hide loading indicator
 */
function hideLoading() {
  const loading = document.getElementById('loading-overlay');

  if (loading) {
    loading.style.display = 'none';
  }
}

/**
 * Show error message
 * @param {string} message - Error message
 */
function showError(message) {
  console.error(message);

  // Create error toast if not exists
  let errorToast = document.getElementById('error-toast');

  if (!errorToast) {
    errorToast = document.createElement('div');
    errorToast.id = 'error-toast';
    errorToast.style.position = 'fixed';
    errorToast.style.bottom = '20px';
    errorToast.style.right = '20px';
    errorToast.style.backgroundColor = '#f44336';
    errorToast.style.color = 'white';
    errorToast.style.padding = '15px';
    errorToast.style.borderRadius = '4px';
    errorToast.style.zIndex = '1000';
    errorToast.style.transition = 'opacity 0.5s';
    errorToast.style.opacity = '0';

    document.body.appendChild(errorToast);
  }

  // Set message
  errorToast.textContent = message;

  // Show toast
  errorToast.style.opacity = '1';

  // Hide after 5 seconds
  setTimeout(() => {
    errorToast.style.opacity = '0';
  }, 5000);
}

/**
 * Update loading progress
 * @param {number} percent - Progress percentage
 */
function updateLoadingProgress(percent) {
  const spinner = document.querySelector('#loading-overlay .spinner');

  if (spinner) {
    spinner.setAttribute('data-progress', `${percent}%`);
  }
}

/**
 * Clean up resources
 */
function dispose() {
  // Remove event listeners
  window.removeEventListener('resize', onWindowResize);

  // Clear models
  clearCurrentModel();

  // Clear texture cache
  if (textureManager) {
    textureManager.clearCache();
  }

  // Dispose renderer
  if (renderer) {
    renderer.dispose();

    // Remove canvas
    const canvas = renderer.domElement;
    if (canvas && canvas.parentNode) {
      canvas.parentNode.removeChild(canvas);
    }
  }

  // Clear references
  scene = null;
  camera = null;
  renderer = null;
  controls = null;
  currentModel = null;
  textureManager = null;
}

// Call dispose when window unloads
window.addEventListener('beforeunload', dispose);

// Export the EnhancedFabricViewer class
export class EnhancedFabricViewer {
  constructor(canvasContainer) {
    // Initialize properties
    this.container = canvasContainer;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.currentModel = null;
    this.currentLights = [];
    this.currentMaterial = null;
    this.isInitialized = false;
    this.loadingManager = new THREE.LoadingManager();
    this.textureLoader = new THREE.TextureLoader(this.loadingManager);

    // Initialize GLTFLoader
    this.gltfLoader = new THREE.GLTFLoader();

    // Setup loading manager events
    this.setupLoadingManager();

    // Initialize the viewer
    this.init();

    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());
  }

  // ... rest of the EnhancedFabricViewer class methods ...
}
