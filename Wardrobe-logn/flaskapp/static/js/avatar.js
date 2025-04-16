// Global variables
let scene, camera, renderer, controls;
let avatarContainer;
let currentAvatar = null;
let currentHair = null;
let currentClothing = {};
let isLoading = false;

// Model paths for different genders
const MODEL_PATHS = {
    male: '/static/models/avatar/male.gltf',
    female: '/static/models/avatar/female.gltf'
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
    initThreeJS();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load wardrobe items
    loadWardrobeItems('tops');
    
    // Try to load existing avatar
    loadExistingAvatar();
});

// Initialize Three.js scene
function initThreeJS() {
    // Get container
    avatarContainer = document.getElementById('avatarContainer');
    
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(
        45, 
        avatarContainer.clientWidth / avatarContainer.clientHeight, 
        0.1, 
        1000
    );
    camera.position.set(0, 1.5, 3);
    camera.lookAt(0, 1, 0);
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(avatarContainer.clientWidth, avatarContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    avatarContainer.appendChild(renderer.domElement);
    
    // Create controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1.5;
    controls.maxDistance = 4;
    controls.maxPolarAngle = Math.PI / 2;
    
    // Add lights
    setupLights();
    
    // Add grid for reference
    const gridHelper = new THREE.GridHelper(10, 10);
    scene.add(gridHelper);
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Set up lights
function setupLights() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // Main directional light
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(5, 5, 5);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 1024;
    mainLight.shadow.mapSize.height = 1024;
    scene.add(mainLight);
    
    // Fill light
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-5, 3, -5);
    scene.add(fillLight);
    
    // Rim light for edge definition
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
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
    camera.aspect = avatarContainer.clientWidth / avatarContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(avatarContainer.clientWidth, avatarContainer.clientHeight);
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
        const avatarModel = await loadAvatarModel(gender);
        
        // Apply skin color
        if (avatarData.skin_color) {
            applySkinColor(avatarModel, avatarData.skin_color);
        }
        
        // Load and apply hair style
        if (avatarData.hair_style) {
            await loadHairStyle(gender, avatarData.hair_style);
            
            // Apply hair color
            if (avatarData.hair_color && currentHair) {
                applyHairColor(currentHair, avatarData.hair_color);
            }
        }
        
        // Update UI controls to match avatar
        updateUIControls(avatarData);
        
        // Update camera position to focus on avatar
        camera.position.set(0, 1.5, 3);
        controls.target.set(0, 1, 0);
        controls.update();
        
        return avatarModel;
    } catch (error) {
        console.error('Error updating avatar:', error);
        throw error;
    }
}

// Load avatar model
async function loadAvatarModel(gender) {
    return new Promise((resolve, reject) => {
        // Get model path
        const modelPath = MODEL_PATHS[gender.toLowerCase()] || MODEL_PATHS.female;
        
        // Create loader
        const loader = new THREE.GLTFLoader();
        
        // Load model
        loader.load(
            modelPath,
            (gltf) => {
                // Remove current avatar if exists
                if (currentAvatar) {
                    scene.remove(currentAvatar);
                }
                
                // Set new avatar
                currentAvatar = gltf.scene;
                
                // Position and scale avatar
                currentAvatar.position.set(0, 0, 0);
                currentAvatar.scale.set(1, 1, 1);
                
                // Setup shadows
                currentAvatar.traverse((node) => {
                    if (node.isMesh) {
                        node.castShadow = true;
                        node.receiveShadow = true;
                    }
                });
                
                // Add to scene
                scene.add(currentAvatar);
                
                resolve(currentAvatar);
            },
            (xhr) => {
                // Progress
                console.log((xhr.loaded / xhr.total * 100) + '% loaded');
            },
            (error) => {
                console.error('Error loading avatar model:', error);
                reject(error);
            }
        );
    });
}

// Load hair style
async function loadHairStyle(gender, style) {
    return new Promise((resolve, reject) => {
        // Remove current hair if exists
        if (currentHair) {
            scene.remove(currentHair);
            currentHair = null;
        }
        
        // Get path for the requested hair style
        const genderHairStyles = HAIR_PATHS[gender.toLowerCase()] || HAIR_PATHS.female;
        const hairPath = genderHairStyles[style] || genderHairStyles.style1;
        
        // Create loader
        const loader = new THREE.GLTFLoader();
        
        // Load hair model
        loader.load(
            hairPath,
            (gltf) => {
                // Set new hair
                currentHair = gltf.scene;
                
                // Position hair relative to avatar
                // This might need adjustment based on your models
                currentHair.position.set(0, 1.65, 0);
                
                // Setup shadows
                currentHair.traverse((node) => {
                    if (node.isMesh) {
                        node.castShadow = true;
                    }
                });
                
                // Add to scene
                scene.add(currentHair);
                
                resolve(currentHair);
            },
            (xhr) => {
                // Progress
                console.log('Hair: ' + (xhr.loaded / xhr.total * 100) + '% loaded');
            },
            (error) => {
                console.error('Error loading hair model:', error);
                reject(error);
            }
        );
    });
}

// Apply skin color to avatar
function applySkinColor(avatarModel, skinColor) {
    if (!avatarModel) return;
    
    avatarModel.traverse((node) => {
        if (node.isMesh) {
            // Check if this is a skin material by name
            const name = node.name.toLowerCase();
            if (name.includes('skin') || name.includes('body') || name.includes('face')) {
                // Apply color
                node.material.color.setRGB(skinColor[0], skinColor[1], skinColor[2]);
                
                // Improve material properties for more realistic skin
                node.material.roughness = 0.7;
                node.material.metalness = 0.0;
            }
        }
    });
}

// Apply hair color
function applyHairColor(hairModel, hairColor) {
    if (!hairModel) return;
    
    hairModel.traverse((node) => {
        if (node.isMesh) {
            // Apply color to all hair meshes
            node.material.color.setRGB(hairColor[0], hairColor[1], hairColor[2]);
            
            // Improve material properties for hair
            node.material.roughness = 0.6;
            node.material.metalness = 0.1;
        }
    });
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

// Create a fallback avatar when no model is available
function createFallbackAvatar(gender = 'female') {
    // Remove current avatar if exists
    if (currentAvatar) {
        scene.remove(currentAvatar);
    }
    
    // Create a group to hold all avatar parts
    const avatar = new THREE.Group();
    
    // Create head (sphere)
    const headGeometry = new THREE.SphereGeometry(0.25, 32, 32);
    const headMaterial = new THREE.MeshStandardMaterial({
        color: 0xFFE0BD,
        roughness: 0.7,
        metalness: 0.0
    });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = 1.7;
    avatar.add(head);
    
    // Create eyes
    const eyeGeometry = new THREE.SphereGeometry(0.03, 16, 16);
    const eyeMaterial = new THREE.MeshStandardMaterial({
        color: 0xFFFFFF,
        roughness: 0.2,
        metalness: 0.0
    });
    
    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    leftEye.position.set(-0.08, 1.7, 0.2);
    avatar.add(leftEye);
    
    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    rightEye.position.set(0.08, 1.7, 0.2);
    avatar.add(rightEye);
    
    // Create pupils
    const pupilGeometry = new THREE.SphereGeometry(0.015, 16, 16);
    const pupilMaterial = new THREE.MeshStandardMaterial({
        color: 0x000000,
        roughness: 0.1,
        metalness: 0.0
    });
    
    const leftPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
    leftPupil.position.set(-0.08, 1.7, 0.225);
    avatar.add(leftPupil);
    
    const rightPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
    rightPupil.position.set(0.08, 1.7, 0.225);
    avatar.add(rightPupil);
    
    // Create neck
    const neckGeometry = new THREE.CylinderGeometry(0.1, 0.12, 0.2, 32);
    const neckMaterial = new THREE.MeshStandardMaterial({
        color: 0xFFE0BD,
        roughness: 0.7,
        metalness: 0.0
    });
    const neck = new THREE.Mesh(neckGeometry, neckMaterial);
    neck.position.y = 1.5;
    avatar.add(neck);
    
    // Create torso
    const torsoGeometry = new THREE.CylinderGeometry(
        gender === 'female' ? 0.25 : 0.3, 
        gender === 'female' ? 0.2 : 0.25, 
        0.6, 
        32
    );
    const torsoMaterial = new THREE.MeshStandardMaterial({
        color: 0x3388cc,  // Blue shirt
        roughness: 0.8,
        metalness: 0.0
    });
    const torso = new THREE.Mesh(torsoGeometry, torsoMaterial);
    torso.position.y = 1.2;
    avatar.add(torso);
    
    // Create legs
    const legGeometry = new THREE.CylinderGeometry(
        0.12, 
        0.1, 
        0.9, 
        32
    );
    const legMaterial = new THREE.MeshStandardMaterial({
        color: 0x222222,  // Dark pants
        roughness: 0.8,
        metalness: 0.0
    });
    
    const leftLeg = new THREE.Mesh(legGeometry, legMaterial);
    leftLeg.position.set(-0.1, 0.5, 0);
    avatar.add(leftLeg);
    
    const rightLeg = new THREE.Mesh(legGeometry, legMaterial);
    rightLeg.position.set(0.1, 0.5, 0);
    avatar.add(rightLeg);
    
    // Create arms
    const armGeometry = new THREE.CylinderGeometry(
        0.07, 
        0.06, 
        0.6, 
        32
    );
    const armMaterial = new THREE.MeshStandardMaterial({
        color: 0xFFE0BD,
        roughness: 0.7,
        metalness: 0.0
    });
    
    const leftArm = new THREE.Mesh(armGeometry, armMaterial);
    leftArm.position.set(-0.35, 1.2, 0);
    leftArm.rotation.z = -0.2;
    avatar.add(leftArm);
    
    const rightArm = new THREE.Mesh(armGeometry, armMaterial);
    rightArm.position.set(0.35, 1.2, 0);
    rightArm.rotation.z = 0.2;
    avatar.add(rightArm);
    
    // Create simple hair
    const hairGeometry = new THREE.SphereGeometry(0.28, 32, 32, 0, Math.PI * 2, 0, Math.PI / 2);
    const hairMaterial = new THREE.MeshStandardMaterial({
        color: 0x4A3728,  // Brown hair
        roughness: 0.6,
        metalness: 0.0
    });
    const hair = new THREE.Mesh(hairGeometry, hairMaterial);
    hair.position.y = 1.75;
    hair.rotation.x = Math.PI;
    avatar.add(hair);
    
    // Add to scene
    scene.add(avatar);
    currentAvatar = avatar;
    
    return avatar;
}