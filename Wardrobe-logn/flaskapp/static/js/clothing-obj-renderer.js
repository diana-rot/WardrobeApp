// Enhanced Clothing Renderer for OBJ Files
class ClothingOBJRenderer {
    constructor(options = {}) {
        console.log('[ClothingOBJRenderer] Initializing...');

        this.scene = options.scene || null;
        this.avatar = options.avatar || null;
        this.currentClothing = new Map(); // Map of itemId -> clothing mesh
        this.debug = options.debug || false;

        // Initialize loaders
        this.objLoader = new THREE.OBJLoader();
        this.textureLoader = new THREE.TextureLoader();
        this.mtlLoader = new THREE.MTLLoader();

        // Clothing categories and their positioning
        this.categoryPositions = {
            'tops': { position: [0, 1.4, 0], scale: [0.8, 0.8, 0.8] },
            'bottoms': { position: [0, 0.6, 0], scale: [0.8, 0.8, 0.8] },
            'dresses': { position: [0, 1.0, 0], scale: [0.8, 0.8, 0.8] },
            'skirts': { position: [0, 0.8, 0], scale: [0.8, 0.8, 0.8] },
            'outerwear': { position: [0, 1.3, 0], scale: [0.85, 0.85, 0.85] },
            'shoes': { position: [0, 0.1, 0], scale: [0.6, 0.6, 0.6] },
            'accessories': { position: [0, 1.5, 0], scale: [0.5, 0.5, 0.5] }
        };

        // Category mapping for Fashion-MNIST and other datasets
        this.categoryMapping = {
            'T-shirt/top': 'tops',
            'Shirt': 'tops',
            'Pullover': 'tops',
            'Top': 'tops',
            'tshirt': 'tops',
            'Trouser': 'bottoms',
            'Pants': 'bottoms',
            'pants': 'bottoms',
            'Jeans': 'bottoms',
            'jeans': 'bottoms',
            'Dress': 'dresses',
            'dress': 'dresses',
            'Skirt': 'skirts',
            'skirt': 'skirts',
            'maxi_skirt': 'skirts',
            'Jacket': 'outerwear',
            'Coat': 'outerwear',
            'Sandal': 'shoes',
            'Sneaker': 'shoes',
            'Ankle boot': 'shoes',
            'Bag': 'accessories'
        };
    }

    // Set scene and avatar references
    setReferences(scene, avatar) {
        this.scene = scene;
        this.avatar = avatar;
        console.log('[ClothingOBJRenderer] References set:', {
            hasScene: !!this.scene,
            hasAvatar: !!this.avatar
        });
    }

    // Main function to handle clothing item clicks
    async handleClothingItemClick(itemElement) {
        try {
            const itemId = itemElement.dataset.itemId;
            const itemType = itemElement.dataset.itemType || itemElement.dataset.category;

            if (!itemId) {
                console.error('[ClothingOBJRenderer] No item ID found');
                return false;
            }

            console.log(`[ClothingOBJRenderer] Handling click for item: ${itemId}, type: ${itemType}`);

            // Toggle selection
            const isSelected = itemElement.classList.contains('selected');

            if (isSelected) {
                // Remove clothing
                await this.removeClothing(itemId);
                itemElement.classList.remove('selected');
                this.showMessage(`Removed ${itemType}`, 'info');
            } else {
                // Load clothing
                const success = await this.loadClothingFromDatabase(itemId);
                if (success) {
                    itemElement.classList.add('selected');
                    this.showMessage(`Applied ${itemType}`, 'success');
                } else {
                    this.showMessage(`Failed to load ${itemType}`, 'error');
                }
            }

            return true;
        } catch (error) {
            console.error('[ClothingOBJRenderer] Error handling click:', error);
            this.showMessage('Error loading clothing item', 'error');
            return false;
        }
    }

    // Load clothing from database using the item ID
    async loadClothingFromDatabase(itemId) {
        if (!this.scene) {
            console.error('[ClothingOBJRenderer] Scene not set');
            return false;
        }

        this.showLoading('Loading clothing...');

        try {
            // Fetch item data from database
            console.log(`[ClothingOBJRenderer] Fetching item data for ID: ${itemId}`);
            const response = await fetch(`/api/wardrobe/item/${itemId}`);

            if (!response.ok) {
                throw new Error(`Failed to fetch item data: ${response.status}`);
            }

            const itemData = await response.json();
            console.log('[ClothingOBJRenderer] Item data received:', itemData);

            if (!itemData.success) {
                throw new Error(itemData.error || 'Failed to load item data');
            }

            // Build the OBJ file path from the generated models
            const objPath = this.buildOBJPath(itemData);

            if (!objPath) {
                console.warn(`[ClothingOBJRenderer] Cannot build OBJ path for item ${itemId}, creating fallback`);
                return await this.createFallbackClothing(itemData);
            }

            console.log(`[ClothingOBJRenderer] Loading OBJ from: ${objPath}`);

            const clothingMesh = await this.loadOBJFile(objPath, itemData);

            if (clothingMesh) {
                // Remove any existing clothing of the same category
                const category = this.getItemCategory(itemData.type || itemData.label);
                await this.removeClothingByCategory(category);

                // Position and add to scene
                this.positionClothing(clothingMesh, category);
                this.scene.add(clothingMesh);

                // Store reference
                this.currentClothing.set(itemId, {
                    mesh: clothingMesh,
                    category: category,
                    itemData: itemData
                });

                console.log(`[ClothingOBJRenderer] Successfully loaded clothing: ${itemId}`);
                return true;
            }

            return false;

        } catch (error) {
            console.error('[ClothingOBJRenderer] Error loading from database:', error);

            // Try to create fallback clothing
            try {
                const fallbackData = {
                    _id: itemId,
                    type: 'unknown',
                    label: 'Clothing Item'
                };
                return await this.createFallbackClothing(fallbackData);
            } catch (fallbackError) {
                console.error('[ClothingOBJRenderer] Fallback creation failed:', fallbackError);
                return false;
            }
        } finally {
            this.hideLoading();
        }
    }

    // Build OBJ file path based on the database item structure
    buildOBJPath(itemData) {
        try {
            // Check if explicit model_3d_path exists
            if (itemData.model_3d_path) {
                return itemData.model_3d_path;
            }

            // Extract user ID from the data
            const userId = itemData.userId || itemData.user_id;
            if (!userId) {
                console.error('[ClothingOBJRenderer] No userId found in item data');
                return null;
            }

            // Get the model task ID
            const modelTaskId = itemData.model_task_id || itemData.modelTaskId;
            if (!modelTaskId) {
                console.error('[ClothingOBJRenderer] No model_task_id found in item data');
                return null;
            }

            // Try different file variations - the system might generate multiple files
            const possibleFiles = [
                `colab_model_task_${modelTaskId}_0.obj`,
                `colab_model_task_${modelTaskId}_1.obj`,
                `colab_model_task_${modelTaskId}_2.obj`,
                `colab_model_task_${modelTaskId}_3.obj`,
                `colab_model_task_${modelTaskId}_4.obj`,
                `colab_model_task_${modelTaskId}.obj`
            ];

            // Build the path with the first possible file
            const basePath = `/static/models/generated/${userId}`;
            const objPath = `${basePath}/${possibleFiles[0]}`;

            console.log(`[ClothingOBJRenderer] Built OBJ path: ${objPath}`);
            console.log(`[ClothingOBJRenderer] Alternative paths available:`, possibleFiles.map(f => `${basePath}/${f}`));

            // Store alternative paths for fallback
            this.alternativePaths = possibleFiles.slice(1).map(f => `${basePath}/${f}`);

            return objPath;

        } catch (error) {
            console.error('[ClothingOBJRenderer] Error building OBJ path:', error);
            return null;
        }
    }

    // Load OBJ file with proper error handling and multiple path attempts
    async loadOBJFile(objPath, itemData) {
        return new Promise((resolve, reject) => {
            console.log(`[ClothingOBJRenderer] Starting OBJ load: ${objPath}`);

            // Try to load the primary path first
            this.attemptOBJLoad(objPath, itemData, resolve, reject, 0);
        });
    }

    // Attempt to load OBJ file with fallback to alternative paths
    attemptOBJLoad(objPath, itemData, resolve, reject, attemptIndex = 0) {
        // Check if there's an associated MTL file
        const mtlPath = objPath.replace('.obj', '.mtl');

        // Try to load MTL first, then OBJ
        this.mtlLoader.load(
            mtlPath,
            (materials) => {
                console.log(`[ClothingOBJRenderer] MTL loaded successfully: ${mtlPath}`);
                materials.preload();
                this.objLoader.setMaterials(materials);
                this.loadOBJWithLoader(objPath, itemData, resolve, reject, attemptIndex);
            },
            (progress) => {
                if (this.debug) {
                    console.log(`[ClothingOBJRenderer] MTL loading progress:`, progress);
                }
            },
            (error) => {
                console.warn(`[ClothingOBJRenderer] MTL not found or failed to load: ${mtlPath}`, error);
                // Continue without materials
                this.loadOBJWithLoader(objPath, itemData, resolve, reject, attemptIndex);
            }
        );
    }

    // Helper function to load OBJ with the loader and handle fallbacks
    loadOBJWithLoader(objPath, itemData, resolve, reject, attemptIndex) {
        this.objLoader.load(
            objPath,
            (object) => {
                console.log(`[ClothingOBJRenderer] OBJ loaded successfully: ${objPath}`);

                // Apply textures and materials
                this.applyClothingTextures(object, itemData);

                // Set up shadows and properties
                object.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;

                        // Ensure material exists
                        if (!child.material) {
                            child.material = new THREE.MeshStandardMaterial({
                                color: 0xcccccc,
                                roughness: 0.7,
                                metalness: 0.1
                            });
                        }
                    }
                });

                // Store metadata
                object.userData = {
                    itemId: itemData._id,
                    type: itemData.type,
                    label: itemData.label,
                    category: this.getItemCategory(itemData.type || itemData.label),
                    loadedFrom: objPath
                };

                resolve(object);
            },
            (progress) => {
                if (this.debug) {
                    const percent = (progress.loaded / progress.total) * 100;
                    console.log(`[ClothingOBJRenderer] OBJ loading: ${percent.toFixed(2)}%`);
                }
            },
            (error) => {
                console.error(`[ClothingOBJRenderer] Failed to load OBJ: ${objPath}`, error);

                // Try alternative paths if available
                if (this.alternativePaths && attemptIndex < this.alternativePaths.length) {
                    const nextPath = this.alternativePaths[attemptIndex];
                    console.log(`[ClothingOBJRenderer] Trying alternative path: ${nextPath}`);
                    this.attemptOBJLoad(nextPath, itemData, resolve, reject, attemptIndex + 1);
                } else {
                    console.error(`[ClothingOBJRenderer] All OBJ loading attempts failed for item: ${itemData._id}`);
                    reject(error);
                }
            }
        );
    }

    // Apply textures and colors to clothing
    applyClothingTextures(object, itemData) {
        try {
            // Apply color if available
            if (itemData.color && itemData.color.rgb) {
                const color = new THREE.Color(
                    itemData.color.rgb[0] / 255,
                    itemData.color.rgb[1] / 255,
                    itemData.color.rgb[2] / 255
                );

                object.traverse((child) => {
                    if (child.isMesh && child.material) {
                        if (Array.isArray(child.material)) {
                            child.material.forEach(mat => {
                                mat.color = color;
                                mat.needsUpdate = true;
                            });
                        } else {
                            child.material.color = color;
                            child.material.needsUpdate = true;
                        }
                    }
                });
            }

            // Apply texture if available
            if (itemData.texture_preview_path) {
                this.textureLoader.load(
                    itemData.texture_preview_path,
                    (texture) => {
                        object.traverse((child) => {
                            if (child.isMesh && child.material) {
                                if (Array.isArray(child.material)) {
                                    child.material.forEach(mat => {
                                        mat.map = texture;
                                        mat.needsUpdate = true;
                                    });
                                } else {
                                    child.material.map = texture;
                                    child.material.needsUpdate = true;
                                }
                            }
                        });
                    },
                    undefined,
                    (error) => {
                        console.warn('[ClothingOBJRenderer] Failed to load texture:', error);
                    }
                );
            }

        } catch (error) {
            console.error('[ClothingOBJRenderer] Error applying textures:', error);
        }
    }

    // Position clothing on avatar
    positionClothing(clothingMesh, category) {
        const position = this.categoryPositions[category] || this.categoryPositions['tops'];

        clothingMesh.position.set(...position.position);
        clothingMesh.scale.set(...position.scale);

        // Additional adjustments based on avatar size
        if (this.avatar) {
            const avatarBox = new THREE.Box3().setFromObject(this.avatar);
            const avatarHeight = avatarBox.max.y - avatarBox.min.y;

            // Scale clothing relative to avatar size
            const scaleFactor = avatarHeight / 1.8; // Assuming 1.8m average height
            clothingMesh.scale.multiplyScalar(scaleFactor);
        }
    }

    // Create fallback clothing when OBJ fails to load
    async createFallbackClothing(itemData) {
        try {
            console.log(`[ClothingOBJRenderer] Creating fallback clothing for: ${itemData._id}`);

            const category = this.getItemCategory(itemData.type || itemData.label);
            const position = this.categoryPositions[category] || this.categoryPositions['tops'];

            let geometry;
            switch (category) {
                case 'tops':
                    geometry = new THREE.CylinderGeometry(0.3, 0.25, 0.5, 16);
                    break;
                case 'bottoms':
                    geometry = new THREE.CylinderGeometry(0.25, 0.2, 0.8, 16);
                    break;
                case 'dresses':
                    geometry = new THREE.CylinderGeometry(0.3, 0.4, 1.2, 16);
                    break;
                case 'shoes':
                    geometry = new THREE.BoxGeometry(0.2, 0.1, 0.3);
                    break;
                default:
                    geometry = new THREE.BoxGeometry(0.3, 0.4, 0.2);
            }

            // Create material with color if available
            let color = 0xcccccc;
            if (itemData.color && itemData.color.rgb) {
                color = new THREE.Color(
                    itemData.color.rgb[0] / 255,
                    itemData.color.rgb[1] / 255,
                    itemData.color.rgb[2] / 255
                );
            }

            const material = new THREE.MeshStandardMaterial({
                color: color,
                roughness: 0.7,
                metalness: 0.1
            });

            const clothingMesh = new THREE.Mesh(geometry, material);
            clothingMesh.position.set(...position.position);
            clothingMesh.scale.set(...position.scale);

            // Set up shadows
            clothingMesh.castShadow = true;
            clothingMesh.receiveShadow = true;

            // Store metadata
            clothingMesh.userData = {
                itemId: itemData._id,
                type: itemData.type,
                label: itemData.label,
                category: category,
                isFallback: true
            };

            // Remove existing clothing of same category
            await this.removeClothingByCategory(category);

            // Add to scene
            this.scene.add(clothingMesh);

            // Store reference
            this.currentClothing.set(itemData._id, {
                mesh: clothingMesh,
                category: category,
                itemData: itemData
            });

            console.log(`[ClothingOBJRenderer] Fallback clothing created for: ${itemData._id}`);
            return true;

        } catch (error) {
            console.error('[ClothingOBJRenderer] Error creating fallback clothing:', error);
            return false;
        }
    }

    // Remove clothing by item ID
    async removeClothing(itemId) {
        if (!this.currentClothing.has(itemId)) {
            console.warn(`[ClothingOBJRenderer] No clothing found with ID: ${itemId}`);
            return false;
        }

        const clothingData = this.currentClothing.get(itemId);
        const mesh = clothingData.mesh;

        // Remove from scene
        this.scene.remove(mesh);

        // Dispose geometry and materials
        mesh.traverse((child) => {
            if (child.isMesh) {
                if (child.geometry) {
                    child.geometry.dispose();
                }
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(mat => {
                            if (mat.map) mat.map.dispose();
                            mat.dispose();
                        });
                    } else {
                        if (child.material.map) child.material.map.dispose();
                        child.material.dispose();
                    }
                }
            }
        });

        // Remove from tracking
        this.currentClothing.delete(itemId);

        console.log(`[ClothingOBJRenderer] Removed clothing: ${itemId}`);
        return true;
    }

    // Remove clothing by category
    async removeClothingByCategory(category) {
        const itemsToRemove = [];

        for (const [itemId, clothingData] of this.currentClothing.entries()) {
            if (clothingData.category === category) {
                itemsToRemove.push(itemId);
            }
        }

        for (const itemId of itemsToRemove) {
            await this.removeClothing(itemId);
        }

        return itemsToRemove.length > 0;
    }

    // Clear all clothing
    async clearAllClothing() {
        const allItems = Array.from(this.currentClothing.keys());

        for (const itemId of allItems) {
            await this.removeClothing(itemId);
        }

        // Also clear UI selections
        document.querySelectorAll('.wardrobe-item.selected').forEach(item => {
            item.classList.remove('selected');
        });

        console.log('[ClothingOBJRenderer] All clothing cleared');
        return true;
    }

    // Get category from item type
    getItemCategory(itemType) {
        if (!itemType) return 'tops';

        const lowerType = itemType.toLowerCase();

        // Direct mapping
        if (this.categoryMapping[lowerType]) {
            return this.categoryMapping[lowerType];
        }

        // Partial matching
        for (const [key, category] of Object.entries(this.categoryMapping)) {
            if (lowerType.includes(key.toLowerCase()) || key.toLowerCase().includes(lowerType)) {
                return category;
            }
        }

        return 'tops'; // Default fallback
    }

    // Utility functions
    showLoading(message = 'Loading...') {
        if (typeof window.showLoading === 'function') {
            window.showLoading(message);
        }
    }

    hideLoading() {
        if (typeof window.hideLoading === 'function') {
            window.hideLoading();
        }
    }

    showMessage(message, type = 'info') {
        if (typeof window.showMessage === 'function') {
            window.showMessage(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Integration with existing system
document.addEventListener('DOMContentLoaded', function() {
    console.log('[ClothingOBJRenderer] Initializing OBJ clothing renderer...');

    // Wait for dependencies
    const initializeRenderer = () => {
        if (typeof THREE !== 'undefined' && THREE.OBJLoader) {
            // Create global instance
            window.clothingOBJRenderer = new ClothingOBJRenderer({
                debug: true
            });

            // Set up event listeners for wardrobe items
            setupOBJClothingEventListeners();

            console.log('[ClothingOBJRenderer] Initialization complete');
        } else {
            console.log('[ClothingOBJRenderer] Waiting for THREE.js and OBJLoader...');
            setTimeout(initializeRenderer, 100);
        }
    };

    initializeRenderer();
});

// Set up event listeners for clothing items
function setupOBJClothingEventListeners() {
    // Update references when avatar or scene changes
    const updateReferences = () => {
        if (window.clothingOBJRenderer) {
            const scene = window.scene || (window.avatarManager && window.avatarManager.scene);
            const avatar = window.avatar || (window.avatarManager && window.avatarManager.avatarModel);

            if (scene) {
                window.clothingOBJRenderer.setReferences(scene, avatar);
            }
        }
    };

    // Initial reference update
    updateReferences();

    // Update references periodically (in case avatar loads after this script)
    const referenceInterval = setInterval(() => {
        updateReferences();

        // Stop checking after 30 seconds
        setTimeout(() => clearInterval(referenceInterval), 30000);
    }, 1000);

    // Handle tab switching
    document.addEventListener('click', function(e) {
        if (e.target.matches('.avatar-tab[data-tab="custom"]')) {
            setTimeout(updateReferences, 100);
        }
    });

    // Handle wardrobe item clicks
    document.addEventListener('click', async function(e) {
        const wardrobeItem = e.target.closest('.wardrobe-item');
        if (!wardrobeItem) return;

        // Check if we're on the custom avatar tab
        const customSection = document.querySelector('#custom-avatar-section');
        const isCustomTabActive = customSection && customSection.classList.contains('active');

        if (isCustomTabActive && window.clothingOBJRenderer) {
            e.preventDefault();
            await window.clothingOBJRenderer.handleClothingItemClick(wardrobeItem);
        }
    });

    // Handle clear outfit button
    const clearButton = document.getElementById('clear-outfit-custom-btn');
    if (clearButton) {
        clearButton.addEventListener('click', async function() {
            if (window.clothingOBJRenderer) {
                await window.clothingOBJRenderer.clearAllClothing();
                window.clothingOBJRenderer.showMessage('Outfit cleared', 'info');
            }
        });
    }
}