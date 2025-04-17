// Global variables
let scene, camera, renderer, controls, avatar;
let avatarContainer;
let currentAvatar = null;
let currentHair = null;
let currentClothing = {};
let isLoading = false;
let loadingOverlay = document.getElementById('loading-overlay');
let loadingProgress = document.getElementById('loading-progress');

// Model paths for different genders
const MODEL_PATHS = {
    male: '/static/models/avatar/male.gltf',
    female: '/static/models/avatar/female_1.glb'
};

// Hair style paths
const HAIR_PATHS = {
    male: {
        style1: '/static/models/avatar/hair/male_short.gltf',
        style2: '/static/models/avatar/hair/male_medium.gltf',
        style3: '/static/models/avatar/hair/male_long.gltf'
    },
    female: {
        style1: '/static/models/avatar/hair/female_short.gltf',
        style2: '/static/models/avatar/hair/female_medium.gltf',
        style3: '/static/models/avatar/hair/female_long.gltf'
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js scene
    init();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load wardrobe items
    loadWardrobeItems('tops');
    
    // Load the female avatar by default
    loadAvatarModel({ gender: 'female' });
});

// Initialize the 3D scene
function init() {
    // Get container
    avatarContainer = document.getElementById('avatarContainer');
    
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Calculate available space for renderer (excluding left menu)
    const leftMenu = document.querySelector('.customization-panel') || document.querySelector('.left-panel');
    const menuWidth = leftMenu ? leftMenu.offsetWidth : 300; // Default to 300px if menu not found
    const availableWidth = window.innerWidth - menuWidth;
    
    // Create camera with better initial position
    camera = new THREE.PerspectiveCamera(45, availableWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 1.6, 3);
    
    // Create renderer with improved settings
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true
    });
    renderer.setSize(availableWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    
    // Position the renderer next to the menu
    avatarContainer.style.position = 'absolute';
    avatarContainer.style.left = menuWidth + 'px';
    avatarContainer.style.top = '0';
    avatarContainer.appendChild(renderer.domElement);
    
    // Add orbit controls with better constraints
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 10;
    controls.target.set(0, 1, 0);
    controls.maxPolarAngle = Math.PI * 0.8;
    controls.minPolarAngle = Math.PI * 0.2;
    
    // Enhanced lighting setup
    setupLighting();
    
    // Add ground plane for better context
    const groundGeometry = new THREE.PlaneGeometry(10, 10);
    const groundMaterial = new THREE.MeshStandardMaterial({ 
        color: 0xcccccc,
        roughness: 0.8,
        metalness: 0.2
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0;
    ground.receiveShadow = true;
    scene.add(ground);
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);
}

// Set up enhanced lighting
function setupLighting() {
    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // Main directional light (sun)
    const mainLight = new THREE.DirectionalLight(0xffffff, 1.0);
    mainLight.position.set(5, 5, 5);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 0.1;
    mainLight.shadow.camera.far = 20;
    mainLight.shadow.camera.left = -5;
    mainLight.shadow.camera.right = 5;
    mainLight.shadow.camera.top = 5;
    mainLight.shadow.camera.bottom = -5;
    mainLight.shadow.bias = -0.0001;
    scene.add(mainLight);
    
    // Fill light from the front
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(-5, 3, 5);
    scene.add(fillLight);
    
    // Rim light from behind
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.2);
    rimLight.position.set(0, 3, -5);
    scene.add(rimLight);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Handle window resize
function onWindowResize() {
    const leftMenu = document.querySelector('.customization-panel') || document.querySelector('.left-panel');
    const menuWidth = leftMenu ? leftMenu.offsetWidth : 300;
    const availableWidth = window.innerWidth - menuWidth;
    
    camera.aspect = availableWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    
    renderer.setSize(availableWidth, window.innerHeight);
    avatarContainer.style.left = menuWidth + 'px';
}

// Set up event listeners
function setupEventListeners() {
    // Avatar form submission
    const avatarForm = document.getElementById('avatarForm');
    if (avatarForm) {
        avatarForm.addEventListener('submit', handleAvatarFormSubmit);
    }
    
    // Update avatar button
    const updateButton = document.getElementById('updateAvatar');
    if (updateButton) {
        updateButton.addEventListener('click', handleAvatarUpdate);
    }
    
    // Clothing category selection
    const categorySelect = document.getElementById('clothingCategory');
    if (categorySelect) {
        categorySelect.addEventListener('change', (e) => {
            loadWardrobeItems(e.target.value);
        });
    }
}

// Handle avatar form submission
async function handleAvatarFormSubmit(e) {
    e.preventDefault();
    
    // Show loading indicator
    showLoading('Generating avatar...');
    
    const formData = new FormData();
    const photoInput = document.getElementById('photo');
    const genderSelect = document.getElementById('gender');
    
    formData.append('photo', photoInput.files[0]);
    formData.append('gender', genderSelect.value);
    
    try {
        const response = await fetch('/api/avatar/generate', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update avatar with the new data
            await updateAvatar(data.avatarData);
        } else {
            alert(data.error || 'Failed to generate avatar');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while generating the avatar');
    } finally {
        // Hide loading indicator
        hideLoading();
    }
}

// Handle manual avatar update
async function handleAvatarUpdate() {
    // Show loading indicator
    showLoading('Updating avatar...');
    
    const skinTone = document.getElementById('skinTone').value;
    const hairColor = document.getElementById('hairColor').value;
    const hairStyle = document.getElementById('hairStyle').value;
    
    // Convert hex color to RGB (0-1 range)
    const skinRGB = hexToRgb(skinTone);
    const hairRGB = hexToRgb(hairColor);
    
    // Create avatar update data
    const avatarData = {
        gender: document.getElementById('gender').value,
        skin_color: [skinRGB.r/255, skinRGB.g/255, skinRGB.b/255],
        hair_color: [hairRGB.r/255, hairRGB.g/255, hairRGB.b/255],
        hair_style: hairStyle
    };
    
    try {
        // Update avatar
        await updateAvatar(avatarData);
        
        // Save changes to server
        await saveAvatarChanges(avatarData);
    } catch (error) {
        console.error('Error updating avatar:', error);
        alert('Error updating avatar');
    } finally {
        // Hide loading indicator
        hideLoading();
    }
}

// Update avatar with new data
async function updateAvatar(avatarData) {
    if (!avatarData) {
        console.error('Invalid avatar data');
        return;
    }
    
    const gender = avatarData.gender || 'female';
    
    try {
        // Load the base avatar model
        const avatarModel = await loadAvatarModel(avatarData);
        
        // Update UI controls to match avatar
        updateUIControls(avatarData);
        
        // Update camera position to focus on avatar
        camera.position.set(0, 1.6, 2);
        controls.target.set(0, 1, 0);
        controls.update();
        
        return avatarModel;
    } catch (error) {
        console.error('Error updating avatar:', error);
        throw error;
    }
}

// Load the avatar model
async function loadAvatarModel(avatarData) {
    showLoadingOverlay();
    
    try {
        // Clean up existing avatar
        if (avatar) {
            scene.remove(avatar);
            avatar.traverse((child) => {
                if (child.isMesh) {
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) {
                        if (Array.isArray(child.material)) {
                            child.material.forEach(material => material.dispose());
                        } else {
                            child.material.dispose();
                        }
                    }
                }
            });
        }

        // Load the model
        const loader = new THREE.GLTFLoader();
        const modelPath = MODEL_PATHS[avatarData?.gender || 'female'];
        console.log('Loading model from:', modelPath);

        const gltf = await new Promise((resolve, reject) => {
            loader.load(
                modelPath,
                resolve,
                (xhr) => {
                    const progress = Math.floor((xhr.loaded / xhr.total) * 100);
                    updateLoadingProgress(progress);
                },
                reject
            );
        });

        // Set up the avatar
        avatar = gltf.scene;
        
        // Scale and position the avatar appropriately
        avatar.scale.set(1, 1, 1);
        avatar.position.set(0, 0, 0);

        // Add the avatar to the scene
        scene.add(avatar);

        // Reset camera position for better view
        camera.position.set(0, 1.6, 2);
        controls.target.set(0, 1, 0);
        controls.update();

        hideLoadingOverlay();
        return true;
    } catch (error) {
        console.error('Error loading avatar model:', error);
        hideLoadingOverlay();
        createFallbackAvatar();
        return false;
    }
}

// Apply facial features to the avatar
function applyFacialFeatures(avatar, features) {
    // Scale face width and height
    const scaleX = features.face_width;
    const scaleY = features.face_height;
    avatar.scale.set(scaleX, scaleY, scaleX);

    // Adjust eye distance
    if (avatar.getObjectByName('eyes')) {
        const eyes = avatar.getObjectByName('eyes');
        eyes.scale.x = features.eye_distance;
    }

    // Adjust nose length
    if (avatar.getObjectByName('nose')) {
        const nose = avatar.getObjectByName('nose');
        nose.scale.y = features.nose_length;
    }

    // Adjust mouth width
    if (avatar.getObjectByName('mouth')) {
        const mouth = avatar.getObjectByName('mouth');
        mouth.scale.x = features.mouth_width;
    }
}

// Create realistic skin material
function createSkinMaterial(skinColor) {
    return new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(skinColor[0]/255, skinColor[1]/255, skinColor[2]/255),
        roughness: 0.3,
        metalness: 0.0,
        clearcoat: 0.1,
        clearcoatRoughness: 0.3,
        sheen: 0.4,
        sheenRoughness: 0.8,
        transmission: 0.2,
        thickness: 0.5
    });
}

// Create realistic hair material
function createHairMaterial(hairColor) {
    return new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(hairColor[0], hairColor[1], hairColor[2]),
        roughness: 0.4,
        metalness: 0.1,
        clearcoat: 0.4,
        clearcoatRoughness: 0.25,
        sheen: 1.0,
        sheenRoughness: 0.3,
        transmission: 0.0
    });
}

// Create a simple fallback avatar
function createFallbackAvatar() {
    const geometry = new THREE.CylinderGeometry(0.3, 0.2, 1.8, 32);
    const material = new THREE.MeshPhongMaterial({ color: 0x808080 });
    avatar = new THREE.Mesh(geometry, material);
    avatar.position.set(0, 0.9, 0);
    scene.add(avatar);
}

// Loading overlay management
function showLoadingOverlay() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
    }
}

function hideLoadingOverlay() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

function updateLoadingProgress(progress) {
    if (loadingProgress) {
        loadingProgress.textContent = `Loading: ${progress}%`;
    }
}

// Try on clothing
async function tryOnClothing(itemId) {
    // Show loading indicator
    showLoading('Trying on item...');
    
    try {
        const response = await fetch('/api/avatar/try-on', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ itemId })
        });
        
        const data = await response.json();
        
        if (data.success && data.item) {
            // Apply the clothing to the avatar
            await applyClothing(data.item);
        } else {
            alert(data.error || 'Failed to try on item');
        }
    } catch (error) {
        console.error('Error trying on clothing:', error);
        alert('An error occurred while trying on the item');
    } finally {
        // Hide loading indicator
        hideLoading();
    }
}

// Apply clothing to avatar
async function applyClothing(item) {
    // This is a simplified version - in a real implementation, 
    // you would load 3D models for each clothing item
    
    // For now, we'll create a simple placeholder representation
    const type = item.type || 'unknown';
    
    // Remove existing clothing of the same type
    if (currentClothing[type]) {
        scene.remove(currentClothing[type]);
        currentClothing[type] = null;
    }
    
    // Parse color
    const color = item.color || '#FFFFFF';
    const colorValue = new THREE.Color(color);
    
    // Create a clothing item based on type
    let clothingMesh;
    
    switch (type) {
        case 'top':
            // Create a simple shirt shape
            const shirtGeo = new THREE.CylinderGeometry(0.3, 0.25, 0.5, 16);
            const shirtMat = new THREE.MeshStandardMaterial({ color: colorValue });
            clothingMesh = new THREE.Mesh(shirtGeo, shirtMat);
            clothingMesh.position.set(0, 1.2, 0);
            break;
            
        case 'bottom':
            // Create simple pants
            const pantsGeo = new THREE.CylinderGeometry(0.25, 0.2, 0.8, 16);
            const pantsMat = new THREE.MeshStandardMaterial({ color: colorValue });
            clothingMesh = new THREE.Mesh(pantsGeo, pantsMat);
            clothingMesh.position.set(0, 0.6, 0);
            break;
            
        case 'dress':
            // Create a simple dress shape
            const dressGeo = new THREE.CylinderGeometry(0.3, 0.4, 1.2, 16);
            const dressMat = new THREE.MeshStandardMaterial({ color: colorValue });
            clothingMesh = new THREE.Mesh(dressGeo, dressMat);
            clothingMesh.position.set(0, 0.8, 0);
            break;
            
        case 'outerwear':
            // Create a simple jacket shape
            const jacketGeo = new THREE.CylinderGeometry(0.35, 0.3, 0.6, 16);
            const jacketMat = new THREE.MeshStandardMaterial({ color: colorValue });
            clothingMesh = new THREE.Mesh(jacketGeo, jacketMat);
            clothingMesh.position.set(0, 1.2, 0);
            break;
            
        case 'shoes':
            // Create simple shoes
            const shoeGeo = new THREE.BoxGeometry(0.2, 0.1, 0.3);
            const shoeMat = new THREE.MeshStandardMaterial({ color: colorValue });
            
            // Create left and right shoes
            const leftShoe = new THREE.Mesh(shoeGeo, shoeMat);
            leftShoe.position.set(-0.15, 0.05, 0);
            
            const rightShoe = new THREE.Mesh(shoeGeo, shoeMat);
            rightShoe.position.set(0.15, 0.05, 0);
            
            // Create a group to hold both shoes
            clothingMesh = new THREE.Group();
            clothingMesh.add(leftShoe);
            clothingMesh.add(rightShoe);
            break;
            
        case 'accessory':
            // Create a simple accessory (like a hat)
            const hatGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.1, 16);
            const hatMat = new THREE.MeshStandardMaterial({ color: colorValue });
            clothingMesh = new THREE.Mesh(hatGeo, hatMat);
            clothingMesh.position.set(0, 1.9, 0);
            break;
            
        default:
            // Generic placeholder
            const geo = new THREE.BoxGeometry(0.3, 0.3, 0.3);
            const mat = new THREE.MeshStandardMaterial({ color: colorValue });
            clothingMesh = new THREE.Mesh(geo, mat);
            clothingMesh.position.set(0, 1.2, 0);
    }
    
    // Add shadow casting
    if (clothingMesh) {
        clothingMesh.traverse((node) => {
            if (node.isMesh) {
                node.castShadow = true;
                node.receiveShadow = true;
            }
        });
        
        // Add to scene
        scene.add(clothingMesh);
        
        // Store for later reference
        currentClothing[type] = clothingMesh;
        
        // Attach item data for reference
        clothingMesh.userData = {
            itemId: item.id,
            label: item.label,
            type: item.type
        };
    }
    
    return clothingMesh;
}

// Load wardrobe items
async function loadWardrobeItems(category) {
    const container = document.getElementById('wardrobeItems');
    
    // Show loading
    container.innerHTML = '<div class="loading-text">Loading your wardrobe...</div>';
    
    try {
        const response = await fetch(`/api/wardrobe-items?category=${category}`);
        const data = await response.json();
        
        // Clear container
        container.innerHTML = '';
        
        // Get items for the selected category
        const items = data[category] || [];
        
        if (items.length === 0) {
            container.innerHTML = '<div class="no-items">No items in this category</div>';
            return;
        }
        
        // Create item elements
        items.forEach(item => {
            const itemElement = document.createElement('div');
            itemElement.className = 'wardrobe-item';
            itemElement.dataset.itemId = item._id;
            
            // Create item content
            itemElement.innerHTML = `
                <img src="${item.file_path}" alt="${item.label}">
                <div class="item-name">${item.label}</div>
            `;
            
            // Add click event
            itemElement.addEventListener('click', () => {
                // Remove selected class from all items
                document.querySelectorAll('.wardrobe-item').forEach(el => {
                    el.classList.remove('selected');
                });
                
                // Add selected class to clicked item
                itemElement.classList.add('selected');
                
                // Try on the item
                tryOnClothing(item._id);
            });
            
            container.appendChild(itemElement);
        });
    } catch (error) {
        console.error('Error loading wardrobe items:', error);
        container.innerHTML = '<div class="error-text">Failed to load items</div>';
    }
}

// Load existing avatar if available
async function loadExistingAvatar() {
    try {
        const response = await fetch('/api/avatar/get');
        const data = await response.json();
        
        if (data.success && data.avatarData) {
            // Update avatar with saved data
            await updateAvatar(data.avatarData);
        }
    } catch (error) {
        console.error('Error loading existing avatar:', error);
    }
}

// Save avatar changes to server
async function saveAvatarChanges(avatarData) {
    try {
        const response = await fetch('/api/avatar/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(avatarData)
        });
        
        const data = await response.json();
        
        if (!data.success) {
            console.error('Error saving avatar changes:', data.error);
        }
    } catch (error) {
        console.error('Error saving avatar changes:', error);
    }
}

// Update UI controls based on avatar data
function updateUIControls(avatarData) {
    // Update gender select
    if (avatarData.gender) {
        const genderSelect = document.getElementById('gender');
        if (genderSelect) {
            genderSelect.value = avatarData.gender;
        }
    }
    
    // Update skin tone color picker
    if (avatarData.skin_color) {
        const skinToneInput = document.getElementById('skinTone');
        if (skinToneInput) {
            const [r, g, b] = avatarData.skin_color;
            skinToneInput.value = rgbToHex(r * 255, g * 255, b * 255);
        }
    }
    
    // Update hair color picker
    if (avatarData.hair_color) {
        const hairColorInput = document.getElementById('hairColor');
        if (hairColorInput) {
            const [r, g, b] = avatarData.hair_color;
            hairColorInput.value = rgbToHex(r * 255, g * 255, b * 255);
        }
    }
    
    // Update hair style select
    if (avatarData.hair_style) {
        const hairStyleSelect = document.getElementById('hairStyle');
        if (hairStyleSelect) {
            hairStyleSelect.value = avatarData.hair_style;
        }
    }
}

// Convert RGB to hex
function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (Math.round(r) << 16) + (Math.round(g) << 8) + Math.round(b)).toString(16).slice(1);
}

// Convert hex color to RGB
function hexToRgb(hex) {
    // Remove # if present
    hex = hex.replace("#", "");
    
    // Parse hex values
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    
    return { r, g, b };
}

// Show loading overlay
function showLoading(message = 'Loading...') {
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    
    if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
    }
    
    if (loadingText) {
        loadingText.textContent = message;
    }
    
    isLoading = true;
}

// Hide loading overlay
function hideLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
    
    isLoading = false;
}

// Take a screenshot of the avatar
function captureAvatarScreenshot() {
    // Store original camera position
    const originalPosition = camera.position.clone();
    const originalTarget = controls.target.clone();
    
    // Position camera for a good shot
    camera.position.set(0, 1.5, 2.5);
    camera.lookAt(0, 1, 0);
    controls.update();
    
    // Render the scene
    renderer.render(scene, camera);
    
    // Capture the canvas content
    const dataURL = renderer.domElement.toDataURL('image/jpeg');
    
    // Restore camera position
    camera.position.copy(originalPosition);
    controls.target.copy(originalTarget);
    controls.update();
    
    return dataURL;
}

// Save avatar screenshot
async function saveAvatarScreenshot() {
    try {
        const screenshot = captureAvatarScreenshot();
        
        const response = await fetch('/api/avatar/save-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: screenshot })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            console.error('Error saving avatar screenshot:', data.error);
        }
    } catch (error) {
        console.error('Error saving avatar screenshot:', error);
    }
}