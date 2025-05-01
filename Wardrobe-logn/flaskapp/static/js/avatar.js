// Global variables
let scene, camera, renderer, controls, avatar;
let avatarContainer;
let currentAvatar = null;
let currentHair = null;
let currentClothing = {};
let isLoading = false;
let loadingOverlay = null;
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

// Add body type presets
const BODY_TYPE_PRESETS = {
    female: {
        hourglass: {
            shoulder_width: 1.1,
            body_height: 1.0,
            hip_width: 1.1,
            waist_width: 0.8,
            bust: 1.1,
            arm_length: 1.0,
            leg_length: 1.0
        },
        pear: {
            shoulder_width: 0.9,
            body_height: 1.0,
            hip_width: 1.2,
            waist_width: 0.9,
            bust: 0.9,
            arm_length: 1.0,
            leg_length: 1.0
        },
        rectangle: {
            shoulder_width: 1.0,
            body_height: 1.0,
            hip_width: 1.0,
            waist_width: 0.95,
            bust: 1.0,
            arm_length: 1.0,
            leg_length: 1.0
        },
        athletic: {
            shoulder_width: 1.2,
            body_height: 1.05,
            hip_width: 1.0,
            waist_width: 0.9,
            bust: 1.0,
            arm_length: 1.1,
            leg_length: 1.1
        }
    },
    male: {
        athletic: {
            shoulder_width: 1.3,
            body_height: 1.05,
            hip_width: 1.0,
            waist_width: 0.9,
            chest: 1.2,
            arm_length: 1.1,
            leg_length: 1.1
        },
        slim: {
            shoulder_width: 1.1,
            body_height: 1.0,
            hip_width: 0.9,
            waist_width: 0.8,
            chest: 1.0,
            arm_length: 1.0,
            leg_length: 1.05
        },
        regular: {
            shoulder_width: 1.2,
            body_height: 1.0,
            hip_width: 1.0,
            waist_width: 1.0,
            chest: 1.1,
            arm_length: 1.0,
            leg_length: 1.0
        },
        broad: {
            shoulder_width: 1.4,
            body_height: 1.0,
            hip_width: 1.1,
            waist_width: 1.1,
            chest: 1.3,
            arm_length: 1.0,
            leg_length: 1.0
        }
    }
};

// Add to the top with other constants
const MORPH_TARGETS = {
    body: {
        'Muscular': 0,
        'Thin': 1,
        'Fat': 2,
        'Belly': 3,
        'BreastSize': 4,
        'Height': 5
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
    // Check if container exists and is properly initialized
    if (!avatarContainer) {
        console.error('Avatar container not found');
        return;
    }

    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Set up camera
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 1.5, 4);

    // Set up renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    // Get menu width for positioning
    const leftMenu = document.querySelector('.customization-panel') || document.querySelector('.left-panel');
    const menuWidth = leftMenu ? leftMenu.offsetWidth : 0;
    
    // Set up container and renderer
    if (avatarContainer) {
        avatarContainer.style.position = 'absolute';
        avatarContainer.style.left = menuWidth + 'px';
        avatarContainer.style.top = '0';
        
        // Set renderer size based on container
        const width = avatarContainer.clientWidth || window.innerWidth - menuWidth;
        const height = avatarContainer.clientHeight || window.innerHeight;
        renderer.setSize(width, height);
        avatarContainer.appendChild(renderer.domElement);
    }

    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 10;
    controls.target.set(0, 1, 0);
    controls.maxPolarAngle = Math.PI * 0.8;
    controls.minPolarAngle = Math.PI * 0.2;

    // Set up lighting
    setupLighting();

    // Add ground plane
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
    
    // Add photo upload handler
    const photoInput = document.getElementById('photo');
    if (photoInput) {
        photoInput.addEventListener('change', handlePhotoUpload);
    }

    // Set up tab switching
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and panels
            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            
            // Add active class to clicked button and corresponding panel
            button.classList.add('active');
            const panelId = `${button.dataset.tab}-panel`;
            document.getElementById(panelId).classList.add('active');
        });
    });

    // Set up measurement sliders
    setupMeasurementSliders();

    // Add preset card click handlers
    const presetCards = document.querySelectorAll('.preset-card');
    presetCards.forEach(card => {
        card.addEventListener('click', () => {
            // Remove selected class from all cards
            presetCards.forEach(c => c.classList.remove('selected'));
            // Add selected class to clicked card
            card.classList.add('selected');
            // Apply the preset
            applyBodyTypePreset(card.dataset.preset);
        });
    });
}

// Set up measurement sliders and their event handlers
function setupMeasurementSliders() {
    const sliders = document.querySelectorAll('.measurement-slider');
    sliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;
        
        // Update value display
        slider.addEventListener('input', (e) => {
            valueDisplay.textContent = e.target.value;
        });
        
        // Update avatar when slider changes
        slider.addEventListener('change', () => {
            updateAvatarFromControls();
        });
    });
}

// Update avatar based on current control values
function updateAvatarFromControls() {
    if (!avatar) return;

    const features = {
        // Face features
        face_width: parseFloat(document.getElementById('faceWidth').value),
        face_height: parseFloat(document.getElementById('faceHeight').value),
        eye_distance: parseFloat(document.getElementById('eyeDistance').value),
        nose_length: parseFloat(document.getElementById('noseLength').value),
        mouth_width: parseFloat(document.getElementById('mouthWidth').value),
        
        // Body features
        body_height: parseFloat(document.getElementById('height').value),
        shoulder_width: parseFloat(document.getElementById('shoulderWidth').value),
        hip_width: parseFloat(document.getElementById('hipWidth').value),
        arm_length: parseFloat(document.getElementById('armLength').value),
        leg_length: parseFloat(document.getElementById('legLength').value)
    };

    applyFeatures(avatar, features);
}

// Update UI controls based on extracted features
function updateUIFromFeatures(features) {
    // Update face controls
    if (features.face) {
        Object.entries(features.face).forEach(([feature, value]) => {
            const control = document.getElementById(`face-${feature}`);
            if (control) {
                control.value = value;
                // Update value display if it exists
                const display = control.nextElementSibling;
                if (display) {
                    display.textContent = value.toFixed(2);
                }
            }
        });
    }

    // Update body controls
    if (features.body) {
        Object.entries(features.body).forEach(([feature, value]) => {
            const control = document.getElementById(`body-${feature}`);
            if (control) {
                control.value = value;
                // Update value display if it exists
                const display = control.nextElementSibling;
                if (display) {
                    display.textContent = value.toFixed(2);
                }
            }
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
    showLoading();
    
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
        
        // Check for morph targets
        avatar.traverse((node) => {
            if (node.isMesh && node.morphTargetDictionary) {
                console.log('Found morph targets:', node.morphTargetDictionary);
                // Store reference to mesh with morph targets
                node.morphTargetInfluences = node.morphTargetInfluences || [];
                setupMorphTargetControls(node);
            }
        });
        
        // Scale and position the avatar appropriately
        avatar.scale.set(1, 1, 1);
        avatar.position.set(0, 0, 0);

        // Add the avatar to the scene
        scene.add(avatar);

        // Reset camera position for better view
        camera.position.set(0, 1.6, 2);
        controls.target.set(0, 1, 0);
        controls.update();

        hideLoading();
        return true;
    } catch (error) {
        console.error('Error loading avatar model:', error);
        hideLoading();
        createFallbackAvatar();
        return false;
    }
}

// Apply facial and body features to the avatar
function applyFeatures(avatar, features) {
    if (!avatar) return;

    // Store original scale for reference
    const originalScale = avatar.scale.clone();

    // Apply body features with bone transformations
    avatar.traverse((child) => {
        if (child.isMesh) {
            const name = child.name.toLowerCase();
            
            // Body scaling
            if (name.includes('torso') || name.includes('chest')) {
                child.scale.x = originalScale.x * (features.shoulder_width || 1.0);
                child.scale.y = originalScale.y * (features.body_height || 1.0);
                if (features.waist_width) {
                    // Apply waist scaling at the bottom of the torso
                    const waistScale = features.waist_width;
                    child.geometry.scale(waistScale, 1, waistScale);
                }
            }
            
            // Hip area
            if (name.includes('hip')) {
                child.scale.x = originalScale.x * (features.hip_width || 1.0);
            }
            
            // Arms
            if (name.includes('arm') || name.includes('shoulder')) {
                if (name.includes('upper')) {
                    child.scale.y = originalScale.y * (features.arm_length || 1.0) * 0.5;
                } else if (name.includes('lower')) {
                    child.scale.y = originalScale.y * (features.arm_length || 1.0) * 0.5;
                }
            }
            
            // Legs
            if (name.includes('leg') || name.includes('thigh')) {
                if (name.includes('upper')) {
                    child.scale.y = originalScale.y * (features.leg_length || 1.0) * 0.5;
                } else if (name.includes('lower')) {
                    child.scale.y = originalScale.y * (features.leg_length || 1.0) * 0.5;
                }
            }
            
            // Face features
            if (name.includes('head') || name.includes('face')) {
                if (features.face_width) child.scale.x = originalScale.x * features.face_width;
                if (features.face_height) child.scale.y = originalScale.y * features.face_height;
            }
            
            // Apply specific face feature scaling
            if (name.includes('eye') && features.eye_distance) {
                child.scale.x = features.eye_distance;
            }
            if (name.includes('nose') && features.nose_length) {
                child.scale.y = features.nose_length;
            }
            if (name.includes('mouth') && features.mouth_width) {
                child.scale.x = features.mouth_width;
            }
        }
    });

    // Update the scene
    renderer.render(scene, camera);
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
function showLoading() {
    if (!loadingOverlay) {
        loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        loadingOverlay.appendChild(spinner);
        document.body.appendChild(loadingOverlay);
    }
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
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
    container.innerHTML = '<div class="loading">Loading your wardrobe...</div>';
    
    try {
        const response = await fetch(`/api/wardrobe-items?category=${category}`);
        if (!response.ok) {
            throw new Error('Failed to load wardrobe items');
        }
        
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
            
            const hasValidImage = item.file_path && 
                               item.file_path !== 'null' && 
                               !item.file_path.includes('undefined');
            
            const itemName = item.label || 'Unnamed Item';
            
            itemElement.innerHTML = `
                <div class="item-image">
                    ${hasValidImage ? 
                        `<img src="${item.file_path}" 
                             alt="${itemName}"
                             onerror="this.parentElement.innerHTML='<div class=\'error-message\'>Image not available</div>'">`
                        : 
                        `<div class="error-message">No image available</div>`
                    }
                </div>
                <div class="item-name">${itemName}</div>
            `;

            itemElement.addEventListener('click', function() {
                container.querySelectorAll('.wardrobe-item').forEach(el => {
                    el.classList.remove('selected');
                });
                this.classList.add('selected');
            });

            container.appendChild(itemElement);
        });
    } catch (error) {
        console.error('Error loading wardrobe items:', error);
        container.innerHTML = '<div class="error">Failed to load items</div>';
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

// Show message to user
function showMessage(text, type = 'info') {
    const message = document.createElement('div');
    message.className = `message ${type}`;
    message.textContent = text;
    document.body.appendChild(message);

    setTimeout(() => {
        message.remove();
    }, 3000);
}

// Apply body type preset
function applyBodyTypePreset(presetName) {
    const gender = document.getElementById('gender').value;
    const preset = BODY_TYPE_PRESETS[gender][presetName];
    
    if (!preset) return;

    // Update UI controls with preset values
    Object.entries(preset).forEach(([key, value]) => {
        const control = document.getElementById(key);
        if (control) {
            control.value = value;
            control.nextElementSibling.textContent = value;
        }
    });

    // Create features object from preset
    const features = {
        ...preset,
        // Keep existing face features
        face_width: parseFloat(document.getElementById('faceWidth').value),
        face_height: parseFloat(document.getElementById('faceHeight').value),
        eye_distance: parseFloat(document.getElementById('eyeDistance').value),
        nose_length: parseFloat(document.getElementById('noseLength').value),
        mouth_width: parseFloat(document.getElementById('mouthWidth').value)
    };

    // Apply features to avatar
    applyFeatures(avatar, features);
}

// Add function to set up morph target controls
function setupMorphTargetControls(mesh) {
    const morphPanel = document.getElementById('morph-controls');
    if (!morphPanel) {
        console.warn('Morph controls panel not found');
        return;
    }

    // Clear existing controls
    morphPanel.innerHTML = '<h3>Shape Customization</h3>';

    // Create sliders for each morph target
    Object.entries(mesh.morphTargetDictionary).forEach(([name, index]) => {
        const controlGroup = document.createElement('div');
        controlGroup.className = 'measurement-group';
        
        const label = document.createElement('label');
        label.textContent = name;
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'measurement-slider';
        slider.min = '0';
        slider.max = '1';
        slider.step = '0.01';
        slider.value = mesh.morphTargetInfluences[index] || 0;
        
        const value = document.createElement('div');
        value.className = 'measurement-value';
        value.textContent = slider.value;
        
        slider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            mesh.morphTargetInfluences[index] = val;
            value.textContent = val.toFixed(2);
            renderer.render(scene, camera);
        });
        
        controlGroup.appendChild(label);
        controlGroup.appendChild(slider);
        controlGroup.appendChild(value);
        morphPanel.appendChild(controlGroup);
    });
}

// Add function to apply morph target presets
function applyMorphPreset(presetName) {
    if (!avatar) return;
    
    avatar.traverse((node) => {
        if (node.isMesh && node.morphTargetDictionary) {
            // Reset all morph targets
            node.morphTargetInfluences.fill(0);
            
            // Apply preset values
            switch(presetName) {
                case 'athletic':
                    node.morphTargetInfluences[MORPH_TARGETS.body.Muscular] = 0.7;
                    node.morphTargetInfluences[MORPH_TARGETS.body.Fat] = 0.1;
                    break;
                case 'curvy':
                    node.morphTargetInfluences[MORPH_TARGETS.body.BreastSize] = 0.6;
                    node.morphTargetInfluences[MORPH_TARGETS.body.Fat] = 0.3;
                    break;
                case 'slim':
                    node.morphTargetInfluences[MORPH_TARGETS.body.Thin] = 0.8;
                    break;
                // Add more presets as needed
            }
            
            // Update UI controls
            updateMorphControlsUI(node);
        }
    });
    
    renderer.render(scene, camera);
}

// Add function to update morph control UI
function updateMorphControlsUI(mesh) {
    const morphPanel = document.getElementById('morph-controls');
    if (!morphPanel) return;
    
    // Update all sliders
    Object.entries(mesh.morphTargetDictionary).forEach(([name, index]) => {
        const slider = morphPanel.querySelector(`input[data-morph="${name}"]`);
        if (slider) {
            slider.value = mesh.morphTargetInfluences[index];
            slider.nextElementSibling.textContent = mesh.morphTargetInfluences[index].toFixed(2);
        }
    });
}

async function handlePhotoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        showLoading();
        const base64Image = await convertToBase64(file);
        
        const response = await fetch('/api/extract-features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        });

        if (!response.ok) {
            throw new Error('Failed to extract features');
        }

        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // Apply extracted features to the avatar
        await applyExtractedFeatures(result.features);
        
        // Update UI controls with extracted values
        updateUIFromFeatures(result.features);
        
        showMessage('Features extracted and applied successfully!', 'success');
    } catch (error) {
        console.error('Error processing photo:', error);
        showMessage(error.message || 'Error processing photo', 'error');
    } finally {
        hideLoading();
    }
}

function convertToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function applyExtractedFeatures(features) {
    if (!avatar) {
        console.error('Avatar not loaded');
        return;
    }

    console.log('Applying extracted features:', features);

    // Apply face features
    if (features.face) {
        await applyFaceFeatures(features.face);
    }

    // Apply body features
    if (features.body) {
        await applyBodyFeatures(features.body);
    }

    // Trigger a render update
    renderer.render(scene, camera);
}

async function applyFaceFeatures(faceFeatures) {
    if (!faceFeatures) return;
    console.log('Applying face features:', faceFeatures);

    // Find face-related meshes
    avatar.traverse((node) => {
        if (node.isMesh) {
            const name = node.name.toLowerCase();

            // Apply face width
            if (name.includes('face') || name.includes('head')) {
                if (faceFeatures.face_width) {
                    node.scale.x = faceFeatures.face_width;
                }
                if (faceFeatures.face_height) {
                    node.scale.y = faceFeatures.face_height;
                }
            }

            // Apply eye distance
            if (name.includes('eye_l')) {
                node.position.x = -faceFeatures.eye_distance / 2;
            }
            if (name.includes('eye_r')) {
                node.position.x = faceFeatures.eye_distance / 2;
            }

            // Apply nose features
            if (name.includes('nose')) {
                node.scale.y = faceFeatures.nose_length;
            }

            // Apply mouth width
            if (name.includes('mouth') || name.includes('lips')) {
                node.scale.x = faceFeatures.mouth_width;
            }

            // Apply jaw width
            if (name.includes('jaw')) {
                node.scale.x = faceFeatures.jaw_width;
            }
        }
    });

    // Update morph targets if available
    const faceMesh = avatar.getObjectByName('face');
    if (faceMesh && faceMesh.morphTargetDictionary) {
        // Map features to morph targets
        const morphTargetMapping = {
            'face_width': 'FaceWidth',
            'face_height': 'FaceHeight',
            'jaw_width': 'JawWidth',
            'nose_length': 'NoseLength'
        };

        for (const [feature, targetName] of Object.entries(morphTargetMapping)) {
            const targetIndex = faceMesh.morphTargetDictionary[targetName];
            if (targetIndex !== undefined && faceFeatures[feature]) {
                faceMesh.morphTargetInfluences[targetIndex] = faceFeatures[feature];
            }
        }
    }
}

async function applyBodyFeatures(bodyFeatures) {
    if (!bodyFeatures) return;
    console.log('Applying body features:', bodyFeatures);

    // Store original proportions
    const originalScales = new Map();
    
    // Find and store original scales of body parts
    avatar.traverse((node) => {
        if (node.isMesh) {
            originalScales.set(node.name, node.scale.clone());
        }
    });

    // Apply measurements to the avatar
    avatar.traverse((node) => {
        if (node.isMesh) {
            const name = node.name.toLowerCase();
            const originalScale = originalScales.get(node.name);

            if (!originalScale) return;

            // Apply shoulder width
            if (name.includes('shoulder') || name.includes('clavicle')) {
                node.scale.x = originalScale.x * (bodyFeatures.shoulder_width || 1);
            }

            // Apply hip width
            if (name.includes('hip') || name.includes('pelvis')) {
                node.scale.x = originalScale.x * (bodyFeatures.hip_width || 1);
            }

            // Apply torso length
            if (name.includes('spine') || name.includes('torso')) {
                node.scale.y = originalScale.y * (bodyFeatures.torso_length || 1);
            }

            // Apply arm length
            if (name.includes('arm') || name.includes('forearm')) {
                node.scale.y = originalScale.y * (bodyFeatures.arm_length || 1);
            }

            // Apply leg length
            if (name.includes('leg') || name.includes('thigh') || name.includes('calf')) {
                node.scale.y = originalScale.y * (bodyFeatures.leg_length || 1);
            }

            // Apply chest width
            if (name.includes('chest') || name.includes('torso')) {
                node.scale.x = originalScale.x * (bodyFeatures.chest_width || 1);
            }
        }
    });

    // Apply morph targets if available
    avatar.traverse((node) => {
        if (node.isMesh && node.morphTargetDictionary) {
            // Map features to morph targets
            const morphTargetMapping = {
                'shoulder_width': 'ShoulderWidth',
                'hip_width': 'HipWidth',
                'torso_length': 'TorsoLength',
                'chest_width': 'ChestWidth'
            };

            for (const [feature, targetName] of Object.entries(morphTargetMapping)) {
                const targetIndex = node.morphTargetDictionary[targetName];
                if (targetIndex !== undefined && bodyFeatures[feature]) {
                    node.morphTargetInfluences[targetIndex] = bodyFeatures[feature];
                }
            }
        }
    });

    // Update skeleton if available
    const skeleton = avatar.getObjectByName('Armature');
    if (skeleton) {
        skeleton.updateMatrixWorld(true);
    }
}

class AvatarManager {
    constructor(options = {}) {
        if (!options.container) {
            console.error('Container element is required for AvatarManager');
            return;
        }
        
        this.container = options.container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.avatarModel = null;
        this.animations = {};
        this.mixer = null;
        
        // Initialize the 3D scene
        this.init();
        
        // Set up event listeners
        this.setupEventListeners();
    }

    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);

        // Set up camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(0, 1.5, 4);

        // Set up renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        // Add renderer to container
        this.container.appendChild(this.renderer.domElement);

        // Set up controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 1.6, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const mainLight = new THREE.DirectionalLight(0xffffff, 1);
        mainLight.position.set(5, 5, 5);
        mainLight.castShadow = true;
        this.scene.add(mainLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 3, -5);
        this.scene.add(fillLight);

        // Add ground plane
        const groundGeometry = new THREE.PlaneGeometry(10, 10);
        const groundMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xcccccc,
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize(), false);

        // Start animation loop
        this.animate();

        // Setup form submission
        const form = document.getElementById('avatar-form');
        form.addEventListener('submit', this.handleFormSubmit.bind(this));
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    showLoading(message = 'Loading...') {
        this.loadingOverlay.style.display = 'flex';
        this.loadingProgress.textContent = message;
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    async handleFormSubmit(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const photo = formData.get('photo');
        const gender = formData.get('gender');

        if (!photo) {
            alert('Please select a photo');
            return;
        }

        this.showLoading('Processing photo...');

        try {
            // First, extract features from the photo
            const featuresFormData = new FormData();
            featuresFormData.append('photo', photo);

            const featuresResponse = await fetch('/api/extract-features', {
                method: 'POST',
                body: featuresFormData
            });

            if (!featuresResponse.ok) {
                throw new Error('Failed to extract features');
            }

            const features = await featuresResponse.json();
            console.log('Features extracted:', features);

            // Then generate the avatar with the extracted features
            const avatarResponse = await fetch('/api/avatar/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    features: features,
                    gender: gender
                })
            });

            if (!avatarResponse.ok) {
                throw new Error('Failed to generate avatar');
            }

            const avatarData = await avatarResponse.json();
            console.log('Avatar generated:', avatarData);

            // Load the generated avatar model
            await this.loadAvatar(avatarData.modelUrl);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your photo');
        } finally {
            this.hideLoading();
        }
    }

    async loadAvatar(url) {
        this.showLoading('Loading avatar...');

        try {
            const loader = new THREE.GLTFLoader();
            const gltf = await loader.loadAsync(url);

            // Remove existing avatar if any
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
            }

            this.avatarModel = gltf.scene;
            this.avatarModel.traverse((node) => {
                if (node.isMesh) {
                    node.castShadow = true;
                    node.receiveShadow = true;
                }
            });

            // Center the avatar
            const box = new THREE.Box3().setFromObject(this.avatarModel);
            const center = box.getCenter(new THREE.Vector3());
            this.avatarModel.position.sub(center);

            // Scale the avatar to a reasonable size
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 1.7 / maxDim;
            this.avatarModel.scale.multiplyScalar(scale);

            this.scene.add(this.avatarModel);

        } catch (error) {
            console.error('Error loading avatar:', error);
            alert('Failed to load avatar model');
        } finally {
            this.hideLoading();
        }
    }

    setupEventListeners() {
        // Avatar form submission
        const avatarForm = document.getElementById('avatar-form');
        if (avatarForm) {
            avatarForm.addEventListener('submit', this.handleFormSubmit.bind(this));
        }
        
        // Update avatar button
        const updateButton = document.getElementById('updateAvatar');
        if (updateButton) {
            updateButton.addEventListener('click', this.handleAvatarUpdate.bind(this));
        }
        
        // Clothing category selection
        const categorySelect = document.getElementById('clothingCategory');
        if (categorySelect) {
            categorySelect.addEventListener('change', (e) => {
                this.loadWardrobeItems(e.target.value);
            });
        }
        
        // Add photo upload handler
        const photoInput = document.getElementById('photo');
        if (photoInput) {
            photoInput.addEventListener('change', this.handlePhotoUpload.bind(this));
        }

        // Set up tab switching
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Remove active class from all buttons and panels
                tabButtons.forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
                
                // Add active class to clicked button and corresponding panel
                button.classList.add('active');
                const panelId = `${button.dataset.tab}-panel`;
                document.getElementById(panelId).classList.add('active');
            });
        });

        // Set up measurement sliders
        this.setupMeasurementSliders();

        // Add preset card click handlers
        const presetCards = document.querySelectorAll('.preset-card');
        presetCards.forEach(card => {
            card.addEventListener('click', () => {
                // Remove selected class from all cards
                presetCards.forEach(c => c.classList.remove('selected'));
                // Add selected class to clicked card
                card.classList.add('selected');
                // Apply the preset
                this.applyBodyTypePreset(card.dataset.preset);
            });
        });
    }

    setupMeasurementSliders() {
        const sliders = document.querySelectorAll('.measurement-slider');
        sliders.forEach(slider => {
            const valueDisplay = slider.nextElementSibling;
            
            // Update value display
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = e.target.value;
            });
            
            // Update avatar when slider changes
            slider.addEventListener('change', () => {
                this.updateAvatarFromControls();
            });
        });
    }

    updateAvatarFromControls() {
        if (!this.avatarModel) return;

        const features = {
            // Face features
            face_width: parseFloat(document.getElementById('faceWidth').value),
            face_height: parseFloat(document.getElementById('faceHeight').value),
            eye_distance: parseFloat(document.getElementById('eyeDistance').value),
            nose_length: parseFloat(document.getElementById('noseLength').value),
            mouth_width: parseFloat(document.getElementById('mouthWidth').value),
            
            // Body features
            body_height: parseFloat(document.getElementById('height').value),
            shoulder_width: parseFloat(document.getElementById('shoulderWidth').value),
            hip_width: parseFloat(document.getElementById('hipWidth').value),
            arm_length: parseFloat(document.getElementById('armLength').value),
            leg_length: parseFloat(document.getElementById('legLength').value)
        };

        this.applyFeatures(this.avatarModel, features);
    }

    applyFeatures(avatar, features) {
        if (!avatar) return;

        // Store original scale for reference
        const originalScale = avatar.scale.clone();

        // Apply body features with bone transformations
        avatar.traverse((child) => {
            if (child.isMesh) {
                const name = child.name.toLowerCase();
                
                // Body scaling
                if (name.includes('torso') || name.includes('chest')) {
                    child.scale.x = originalScale.x * (features.shoulder_width || 1.0);
                    child.scale.y = originalScale.y * (features.body_height || 1.0);
                    if (features.waist_width) {
                        // Apply waist scaling at the bottom of the torso
                        const waistScale = features.waist_width;
                        child.geometry.scale(waistScale, 1, waistScale);
                    }
                }
                
                // Hip area
                if (name.includes('hip')) {
                    child.scale.x = originalScale.x * (features.hip_width || 1.0);
                }
                
                // Arms
                if (name.includes('arm') || name.includes('shoulder')) {
                    if (name.includes('upper')) {
                        child.scale.y = originalScale.y * (features.arm_length || 1.0) * 0.5;
                    } else if (name.includes('lower')) {
                        child.scale.y = originalScale.y * (features.arm_length || 1.0) * 0.5;
                    }
                }
                
                // Legs
                if (name.includes('leg') || name.includes('thigh')) {
                    if (name.includes('upper')) {
                        child.scale.y = originalScale.y * (features.leg_length || 1.0) * 0.5;
                    } else if (name.includes('lower')) {
                        child.scale.y = originalScale.y * (features.leg_length || 1.0) * 0.5;
                    }
                }
                
                // Face features
                if (name.includes('head') || name.includes('face')) {
                    if (features.face_width) child.scale.x = originalScale.x * features.face_width;
                    if (features.face_height) child.scale.y = originalScale.y * features.face_height;
                }
                
                // Apply specific face feature scaling
                if (name.includes('eye') && features.eye_distance) {
                    child.scale.x = features.eye_distance;
                }
                if (name.includes('nose') && features.nose_length) {
                    child.scale.y = features.nose_length;
                }
                if (name.includes('mouth') && features.mouth_width) {
                    child.scale.x = features.mouth_width;
                }
            }
        });

        // Update the scene
        this.renderer.render(this.scene, this.camera);
    }

    applyBodyTypePreset(presetName) {
        const gender = document.getElementById('gender').value;
        const preset = BODY_TYPE_PRESETS[gender][presetName];
        
        if (!preset) return;

        // Update UI controls with preset values
        Object.entries(preset).forEach(([key, value]) => {
            const control = document.getElementById(key);
            if (control) {
                control.value = value;
                control.nextElementSibling.textContent = value;
            }
        });

        // Create features object from preset
        const features = {
            ...preset,
            // Keep existing face features
            face_width: parseFloat(document.getElementById('faceWidth').value),
            face_height: parseFloat(document.getElementById('faceHeight').value),
            eye_distance: parseFloat(document.getElementById('eyeDistance').value),
            nose_length: parseFloat(document.getElementById('noseLength').value),
            mouth_width: parseFloat(document.getElementById('mouthWidth').value)
        };

        // Apply features to avatar
        this.applyFeatures(this.avatarModel, features);
    }

    handleAvatarUpdate() {
        // Implementation needed
    }

    handlePhotoUpload(event) {
        // Implementation needed
    }

    loadWardrobeItems(category) {
        // Implementation needed
    }
}