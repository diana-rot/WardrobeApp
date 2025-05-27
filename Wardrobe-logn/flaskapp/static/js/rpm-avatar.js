// Ready Player Me Avatar Integration
class RPMAvatarManager {
    constructor(options = {}) {
        this.container = options.container;
        this.onAvatarLoaded = options.onAvatarLoaded || (() => {});
        this.onAvatarError = options.onAvatarError || (() => {});
        
        this.avatarUrl = null;
        this.avatarModel = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.isLoading = false;
        this.debug = false;

        // Initialize clothingItems map
        this.clothingItems = new Map();

        // Try to load saved avatar on initialization
        this.loadSavedAvatar();

        // Update modelMapping and categoryMapping
        this.modelMapping = {
            // Tops
            'T-shirt/top': '/static/models/clothing/top.glb',
            'Shirt': '/static/models/clothing/top.glb',
            'Pullover': '/static/models/clothing/top.glb',
            'Top': '/static/models/clothing/top.glb',
            'tshirt': '/static/models/clothing/top.glb',
            // Bottoms
            'Trouser': '/static/models/clothing/pants.glb',
            'Pants': '/static/models/clothing/pants.glb',
            'pants': '/static/models/clothing/pants.glb',
            'Jeans': '/static/models/clothing/jeans.glb',
            'jeans': '/static/models/clothing/jeans.glb',
            // Dresses
            'Dress': '/static/models/clothing/dress.glb',
            'dress': '/static/models/clothing/dress.glb',
            // Skirts
            'Skirt': '/static/models/clothing/skirt.glb',
            'skirt': '/static/models/clothing/skirt.glb',
            'maxi_skirt': '/static/models/clothing/maxi_skirt.glb',
            // Outerwear
            'Jacket': '/static/models/clothing/jacket.glb',
            'Coat': '/static/models/clothing/coat.glb',
            // Shoes
            'Sandal': '/static/models/clothing/sandal.glb',
            'Sneaker': '/static/models/clothing/sneaker.glb',
            'Ankle boot': '/static/models/clothing/ankle_boot.glb',
            // Accessories
            'Bag': '/static/models/clothing/bag.glb'
        };
        this.categoryMapping = {
            'tops': ['T-shirt/top', 'Shirt', 'Pullover', 'Top', 'tshirt'],
            'bottoms': ['Trouser', 'Pants', 'pants', 'Jeans', 'jeans'],
            'dresses': ['Dress', 'dress'],
            'skirts': ['Skirt', 'skirt', 'maxi_skirt'],
            'outerwear': ['Jacket', 'Coat'],
            'shoes': ['Sandal', 'Sneaker', 'Ankle boot'],
            'accessories': ['Bag']
        };
    }

    initThreeJS() {
        if (this.scene) {
            // Already initialized
            return;
        }

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);

        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            35,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0.5, 0);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.shadowMap.enabled = true;
        
        // Ensure GLTFLoader is initialized
        this.gltfLoader = new THREE.GLTFLoader();

        // Clear any existing canvas
        while (this.container.firstChild) {
            this.container.removeChild(this.container.firstChild);
        }
        this.container.appendChild(this.renderer.domElement);

        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 1.2;
        this.controls.maxDistance = 4;
        this.controls.minPolarAngle = Math.PI / 4;
        this.controls.maxPolarAngle = Math.PI / 2;
        this.controls.target.set(0, -2.0, 0);
        
        // Setup lighting
        this.setupLighting();

        // Add ground plane

        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start animation loop
        this.animate();
    }
    
    setupLighting() {
        // Clear existing lights
        this.scene.traverse((child) => {
            if (child.isLight) {
                this.scene.remove(child);
            }
        });

        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Directional light (key light)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(2, 2, 2);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);

        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-2, 1, -1);
        this.scene.add(fillLight);

        // Back light
        const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
        backLight.position.set(0, 1, -2);
        this.scene.add(backLight);
    }
    
    addGroundPlane() {
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
        this.scene.add(ground);
    }

    onWindowResize() {
        if (!this.camera || !this.renderer) return;
        
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        if (!this.renderer || !this.scene || !this.camera) return;
        
        requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    async loadSavedAvatar() {
        try {
            const response = await fetch('/api/rpm/get-avatar');
            if (!response.ok) {
                throw new Error('Failed to load saved avatar');
            }
            
            const data = await response.json();
            if (data.avatarUrl) {
                // Load the saved avatar
                await this.loadAvatar(data.avatarUrl);
            }
        } catch (error) {
            console.warn('No saved avatar found:', error);
        }
    }

    async saveAvatarUrl(avatarUrl) {
        try {
            const response = await fetch('/api/rpm/save-avatar', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ avatarUrl })
            });
            
            if (!response.ok) {
                throw new Error('Failed to save avatar URL');
            }
            
            const data = await response.json();
            return data.avatarUrl || avatarUrl;
        } catch (error) {
            console.error('Error saving avatar URL:', error);
            throw error;
        }
    }
    
    showLoadingOverlay(message) {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }
    
    hideLoadingOverlay() {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    updateLoadingProgress(percent) {
        const progress = document.querySelector('.loading-progress');
        if (progress) {
            progress.textContent = `Loading: ${percent}%`;
        }
    }

    async loadAvatar(avatarUrl) {
        if (!avatarUrl) {
            throw new Error('Avatar URL is required');
        }

        this.isLoading = true;
        this.showLoadingOverlay('Loading avatar...');

        try {
            // Convert avatar URL to GLB if necessary
            const glbUrl = this.getGLBUrlFromAvatarUrl(avatarUrl);

            // Remove all previous avatars from the scene (robust cleanup)
            if (this.scene) {
                this.scene.children
                    .filter(obj => obj !== this.currentClothing && (obj.isMesh || obj.type === 'Group'))
                    .forEach(obj => {
                        this.scene.remove(obj);
                        if (obj.traverse) {
                            obj.traverse((node) => {
                                if (node.isMesh) {
                                    if (node.material) node.material.dispose();
                                    if (node.geometry) node.geometry.dispose();
                                }
                            });
                        }
                    });
                this.avatarModel = null;
            }
            // Remove and dispose of all clothing
            if (this.clothingItems && this.clothingItems.size > 0) {
                for (const [itemId, item] of this.clothingItems.entries()) {
                    if (item.mesh) {
                        this.scene.remove(item.mesh);
                        item.mesh.traverse((node) => {
                            if (node.isMesh) {
                                if (node.material) node.material.dispose();
                                if (node.geometry) node.geometry.dispose();
                            }
                        });
                    }
                }
                this.clothingItems.clear();
            }
            if (this.currentClothing) {
                this.scene.remove(this.currentClothing);
                this.currentClothing = null;
            }

            // Initialize Three.js scene if not already done
            if (!this.scene) {
                this.initThreeJS();
            }

            // Load the new avatar
            const loader = new THREE.GLTFLoader();
            loader.setPath(glbUrl);

            const onProgress = (event) => {
                if (event.lengthComputable) {
                    const percent = Math.round((event.loaded / event.total) * 100);
                    this.updateLoadingProgress(percent);
                }
            };

            const gltf = await new Promise((resolve, reject) => {
                loader.load(
                    '',
                    resolve,
                    onProgress,
                    reject
                );
            });
            
            this.avatarModel = gltf.scene;
            // Scale the avatar to make it a little bigger
            this.avatarModel.scale.set(0.4, 0.4, 0.4); // Set scale to 40% of original size
            
            this.avatarModel.traverse((node) => {
                if (node.isMesh) {
                    node.castShadow = true;
                    node.receiveShadow = true;
                    // Enable better material quality
                    if (node.material) {
                        node.material.envMapIntensity = 1.5;
                        node.material.needsUpdate = true;
                    }
                }
            });
            
            this.scene.add(this.avatarModel);
            this.centerCameraOnAvatar();
            
            // Store the URL
            this.avatarUrl = avatarUrl;
            
            // Call success callback
            if (this.onAvatarLoaded) {
                this.onAvatarLoaded(this.avatarModel);
            }
            
        } catch (error) {
            console.error('Error loading avatar:', error);
            if (this.onAvatarError) {
                this.onAvatarError(error);
            }
            throw error;
        } finally {
            this.isLoading = false;
            this.hideLoadingOverlay();
        }
    }

    // Load clothing onto the avatar
    async loadClothing(itemId, imagePath, itemType) {
        if (!this.avatarModel) {
            console.error('No avatar loaded. Please load an avatar first.');
            return false;
        }
        // Defensive: ensure gltfLoader is initialized
        if (!this.gltfLoader) {
            this.gltfLoader = new THREE.GLTFLoader();
        }
        // Defensive: ensure positionAdjustments is initialized
        if (!this.positionAdjustments) {
            this.positionAdjustments = {
                'tops': { y: 1.4, scale: 0.5 },
                'bottoms': { y: 0.6, scale: 0.5 },
                'dresses': { y: 1.0, scale: 0.6 },
                'skirts': { y: 0.7, scale: 0.5 },
                'outerwear': { y: 1.3, scale: 0.55 },
                'shoes': { y: 0.1, scale: 0.4 },
                'accessories': { y: 1.5, scale: 0.4 }
            };
        }
        try {
            console.log('--- [Wardrobe Debug] ---');
            console.log('Loading clothing item:', { itemId, imagePath, itemType });
            // Determine the category of the clothing item
            const category = this.getClothingCategory(itemType);
            console.log(`Mapped itemType "${itemType}" to category: "${category}"`);
            // Get the model path based on item type
            const modelPath = this.getModelPath(itemType);
            console.log(`Model path for "${itemType}": ${modelPath}`);
            // Remove existing clothing of the same category
            await this.removeClothingByCategory(category);
            // Load the model
            const gltf = await new Promise((resolve, reject) => {
                this.gltfLoader.load(
                    modelPath,
                    resolve,
                    (xhr) => {
                        const percent = Math.round((xhr.loaded / xhr.total) * 100);
                        console.log(`[${itemType}] Loading model: ${percent}%`);
                    },
                    reject
                );
            });
            let model = gltf.scene;
            if (!model.position || !model.scale) {
                // Try to find the first mesh or group with position/scale
                model = model.children.find(child => child.position && child.scale) || model;
            }
            // Get position and scale adjustments for this category
            const adjustment = this.positionAdjustments[category] || { y: 1.2, scale: 0.5 };
            console.log(`Position/scale for category "${category}":`, adjustment);
            // Defensive: only set position/scale if available
            if (model && model.position && model.scale) {
                model.position.set(0, adjustment.y, 0);
                model.scale.set(adjustment.scale, adjustment.scale, adjustment.scale);
            } else {
                console.error('Model is missing position or scale property:', model);
                return false;
            }
            // Add to avatar model
            this.avatarModel.add(model);
            console.log(`[DEBUG] ${itemType} added as child of avatarModel`);
            // Store reference
            this.clothingItems.set(itemId, {
                mesh: model,
                type: itemType,
                category: category
            });
            return true;
        } catch (error) {
            console.error('Error loading clothing:', error);
            return false;
        } finally {
            this.hideLoadingOverlay();
        }
    }

    // Add this method to toggle debug visualization
    toggleDebug() {
        this.debug = !this.debug;
        if (this.currentClothing && this.debug) {
            const boxHelper = new THREE.BoxHelper(this.currentClothing, 0xff0000);
            this.scene.add(boxHelper);
            
            const axesHelper = new THREE.AxesHelper(1);
            this.currentClothing.add(axesHelper);
        }
    }

    // Remove clothing item
    removeClothing(itemId) {
        if (!this.clothingItems || !this.clothingItems.has(itemId)) {
            return false;
        }

        const item = this.clothingItems.get(itemId);
        if (item.mesh) {
            this.scene.remove(item.mesh);

            // Clean up resources
            if (item.mesh.material) {
                if (item.mesh.material.map) {
                    item.mesh.material.map.dispose();
                }
                item.mesh.material.dispose();
            }
            if (item.mesh.geometry) {
                item.mesh.geometry.dispose();
            }
        }

        this.clothingItems.delete(itemId);
        return true;
    }

    // Add missing removeClothingByCategory method
    removeClothingByCategory(category) {
        if (!this.clothingItems) return;
        // Find all items in the given category and remove them
        const itemsToRemove = [];
        for (const [itemId, item] of this.clothingItems.entries()) {
            if (item.category === category) {
                itemsToRemove.push(itemId);
            }
        }
        for (const itemId of itemsToRemove) {
            this.removeClothing(itemId);
        }
    }

    // Clear all clothing
    clearAllClothing() {
        if (!this.clothingItems) {
            console.warn('[DEBUG] No clothingItems map found.');
            return;
        }

        console.log('[DEBUG] Clearing all clothing items...');
        for (const [itemId, item] of this.clothingItems.entries()) {
            if (item.mesh) {
                console.log(`[DEBUG] Removing mesh for itemId: ${itemId}`, item.mesh);

                // Remove from any parent
                if (item.mesh.parent) {
                    item.mesh.parent.remove(item.mesh);
                    console.log(`[DEBUG] Removed mesh from parent for itemId: ${itemId}`);
                } else {
                    console.warn(`[DEBUG] Mesh for itemId: ${itemId} had no parent`);
                }

                // Dispose geometry and material
                item.mesh.traverse((node) => {
                    if (node.isMesh) {
                        if (node.geometry) {
                            node.geometry.dispose();
                            console.log(`[DEBUG] Disposed geometry for itemId: ${itemId}`);
                        }
                        if (node.material) {
                            if (Array.isArray(node.material)) {
                                node.material.forEach(mat => {
                                    if (mat.map) mat.map.dispose();
                                    mat.dispose();
                                });
                            } else {
                                if (node.material.map) node.material.map.dispose();
                                node.material.dispose();
                            }
                            console.log(`[DEBUG] Disposed material for itemId: ${itemId}`);
                        }
                    }
                });
            } else {
                console.warn(`[DEBUG] No mesh found for itemId: ${itemId}`);
            }
        }

        this.clothingItems.clear();
        console.log('[DEBUG] clothingItems map cleared.');

        // Deselect all wardrobe items in the UI
        document.querySelectorAll('.wardrobe-item.selected').forEach(item => item.classList.remove('selected'));
        console.log('[DEBUG] Deselected all wardrobe items in UI.');

        // Force a render update
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
            console.log('[DEBUG] Forced scene render after clearing clothing.');
        }
    }

    // Add missing getClothingCategory method
    getClothingCategory(itemType) {
        if (!itemType) return 'tops'; // Default to tops
        const lowerItemType = itemType.toLowerCase();
        for (const [category, types] of Object.entries(this.categoryMapping)) {
            for (const type of types) {
                if (type.toLowerCase() === lowerItemType || lowerItemType.includes(type.toLowerCase())) {
                    return category;
                }
            }
        }
        return 'tops';
    }

    // Add missing getModelPath method
    getModelPath(itemType) {
        if (!itemType) return '/static/models/clothing/top.glb'; // Default
        // First try exact match
        if (this.modelMapping[itemType]) {
            return this.modelMapping[itemType];
        }
        // Try case-insensitive match
        const lowerItemType = itemType.toLowerCase();
        for (const [type, path] of Object.entries(this.modelMapping)) {
            if (type.toLowerCase() === lowerItemType) {
                return path;
            }
        }
        // Try partial match
        for (const [type, path] of Object.entries(this.modelMapping)) {
            if (lowerItemType.includes(type.toLowerCase()) || type.toLowerCase().includes(lowerItemType)) {
                return path;
            }
        }
        // Default to top.glb
        return '/static/models/clothing/top.glb';
    }

    getGLBUrlFromAvatarUrl(avatarUrl) {
        // If URL is already a GLB, return as is
        if (avatarUrl.endsWith('.glb')) {
            return avatarUrl;
        }
        
        // Handle various RPM URL formats
        if (avatarUrl.includes('readyplayer.me')) {
            // Ensure we're requesting a GLB file
            if (avatarUrl.includes('?')) {
                return avatarUrl + '&format=glb';
            } else {
                return avatarUrl + '?format=glb';
            }
        }
        
        // Convert other URLs to GLB
        return avatarUrl.replace('.gltf', '.glb');
    }
    
    centerCameraOnAvatar() {
        if (!this.avatarModel || !this.camera) return;

        // Set initial camera position
        this.camera.position.set(0, 0, 0.5);
        this.controls.target.set(0,0, 0);
        
        // Set camera limits
        this.controls.minDistance = 3.0;
        this.controls.maxDistance = 6.0;
        this.controls.minPolarAngle = Math.PI / 4;
        this.controls.maxPolarAngle = Math.PI / 2;
        
        // Update camera and controls
        this.camera.updateProjectionMatrix();
        this.controls.update();
    }
}

// Export the class
window.RPMAvatarManager = RPMAvatarManager; 