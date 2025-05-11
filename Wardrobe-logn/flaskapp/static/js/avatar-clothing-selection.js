// avatar-clothing-selection.js - Handles clothing selection for custom avatar

class AvatarClothingManager {
    constructor(options = {}) {
        console.log('[AvatarClothingManager] Initializing with options:', options);
        this.scene = options.scene || null;
        this.avatar = options.avatar || null;
        this.currentClothing = {}; // Map of category -> clothing mesh
        this.clothingMapper = options.clothingMapper || new ClothingMapper();

        // Set default paths for different clothing types
        this.clothingPaths = {
            tops: '/static/models/clothing/top.glb',
            bottoms: '/static/models/clothing/bottom.glb',
            dresses: '/static/models/clothing/dress.glb',
            outerwear: '/static/models/clothing/coat.glb',
            shoes: '/static/models/clothing/shoes.glb',
            accessories: '/static/models/clothing/accessory.glb'
        };

        // Override with any provided paths
        if (options.clothingPaths) {
            this.clothingPaths = {...this.clothingPaths, ...options.clothingPaths};
        }

        // Loader for GLB/GLTF models
        this.loader = new THREE.GLTFLoader();

        // Debug mode for additional logging
        this.debug = options.debug || false;
    }

    // Set the avatar and scene references
    setReferences(avatar, scene) {
        this.avatar = avatar;
        this.scene = scene;
        console.log('[AvatarClothingManager] References updated:', {
            hasAvatar: !!this.avatar,
            hasScene: !!this.scene
        });
    }

    // Handle clothing item click
    async handleClothingItemClick(item) {
        if (!item) {
            console.error('[AvatarClothingManager] No item provided');
            return false;
        }

        try {
            if (this.debug) {
                console.log('[AvatarClothingManager] Handling clothing item click:', item);
            }

            // Extract necessary information
            const itemId = item.dataset.itemId || item.id;
            const itemType = item.dataset.itemType || item.dataset.category || 'tops';
            const imageUrl = item.dataset.imageUrl || item.querySelector('img')?.src;

            if (this.debug) {
                console.log('[AvatarClothingManager] Item details:', {
                    itemId,
                    itemType,
                    imageUrl
                });
            }

            // Apply the clothing to the avatar
            const result = await this.applyClothing({
                id: itemId,
                type: itemType,
                label: item.dataset.label || item.querySelector('.item-name')?.textContent || 'Item',
                file_path: imageUrl
            });

            if (result) {
                this.showMessage(`${itemType} applied successfully!`, 'success');
                return true;
            } else {
                this.showMessage(`Failed to apply ${itemType}`, 'error');
                return false;
            }
        } catch (error) {
            console.error('[AvatarClothingManager] Error handling clothing item click:', error);
            this.showMessage('Error applying clothing item', 'error');
            return false;
        }
    }

    // Apply clothing to avatar
    async applyClothing(item) {
        if (!this.scene || !this.avatar) {
            console.error('[AvatarClothingManager] Scene or avatar not initialized');
            return false;
        }

        if (this.debug) {
            console.log('[AvatarClothingManager] Applying clothing:', item);
        }

        try {
            const itemType = item.type.toLowerCase();
            const category = this.getCategory(itemType, item.label);

            // Remove any existing clothing in this category
            this.removeClothing(category);

            // Get the appropriate model path
            const modelPath = this.getModelPath(category);

            if (this.debug) {
                console.log(`[AvatarClothingManager] Loading model from: ${modelPath}`);
            }

            // Load the model
            const showLoading = typeof window.showLoading === 'function' ? window.showLoading : () => {};
            const hideLoading = typeof window.hideLoading === 'function' ? window.hideLoading : () => {};

            showLoading(`Loading ${category}...`);

            try {
                const gltf = await this.loadModel(modelPath);
                const clothingModel = gltf.scene;

                // Apply material/texture based on item color if available
                this.applyTextureToModel(clothingModel, item);

                // Set up the model
                clothingModel.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;

                        // Ensure material is properly initialized
                        if (child.material) {
                            child.material.needsUpdate = true;
                        }
                    }
                });

                // Position the clothing on the avatar
                this.positionClothingOnAvatar(clothingModel, category);

                // Add to scene
                this.scene.add(clothingModel);

                // Store reference
                this.currentClothing[category] = clothingModel;

                // Store metadata
                clothingModel.userData = {
                    itemId: item.id,
                    type: category,
                    label: item.label || category
                };

                // Update avatar preview if the function exists
                if (typeof window.updateAvatarPreview === 'function') {
                    window.updateAvatarPreview();
                }

                hideLoading();
                return true;
            } catch (error) {
                console.error(`[AvatarClothingManager] Error loading model from ${modelPath}:`, error);
                hideLoading();

                // Create a fallback basic shape
                return this.createFallbackClothing(category, item);
            }
        } catch (error) {
            console.error('[AvatarClothingManager] Error applying clothing:', error);
            return false;
        }
    }

    // Load model with promise
    loadModel(modelPath) {
        return new Promise((resolve, reject) => {
            this.loader.load(
                modelPath,
                resolve,
                (xhr) => {
                    const progress = Math.floor((xhr.loaded / xhr.total) * 100);
                    if (this.debug) {
                        console.log(`[AvatarClothingManager] Loading: ${progress}%`);
                    }
                },
                reject
            );
        });
    }

    // Create fallback clothing when model loading fails
    createFallbackClothing(category, item) {
        try {
            if (this.debug) {
                console.log('[AvatarClothingManager] Creating fallback clothing for:', category);
            }

            let geometry, material, position, scale;

            // Define dimensions based on category
            switch (category) {
                case 'tops':
                    geometry = new THREE.CylinderGeometry(0.3, 0.25, 0.5, 16);
                    position = {x: 0, y: 0.8, z: 0};
                    break;
                case 'bottoms':
                    geometry = new THREE.CylinderGeometry(0.25, 0.2, 0.8, 16);
                    position = {x: 0, y: 0.3, z: 0};
                    break;
                case 'dresses':
                    geometry = new THREE.CylinderGeometry(0.3, 0.4, 1.2, 16);
                    position = {x: 0, y: 0.5, z: 0};
                    break;
                case 'outerwear':
                    geometry = new THREE.CylinderGeometry(0.35, 0.3, 0.6, 16);
                    position = {x: 0, y: 0.8, z: 0};
                    break;
                case 'shoes':
                    geometry = new THREE.BoxGeometry(0.2, 0.1, 0.3);
                    position = {x: 0, y: 0.05, z: 0};
                    break;
                case 'accessories':
                    geometry = new THREE.SphereGeometry(0.15, 16, 16);
                    position = {x: 0.3, y: 1.0, z: 0.2};
                    break;
                default:
                    geometry = new THREE.BoxGeometry(0.3, 0.3, 0.3);
                    position = {x: 0, y: 0.6, z: 0};
            }

            // Create colored material based on item
            const color = this.getItemColor(item) || new THREE.Color(0x3498db);
            material = new THREE.MeshStandardMaterial({
                color: color,
                roughness: 0.7,
                metalness: 0.2
            });

            // Create the mesh
            const clothingMesh = new THREE.Mesh(geometry, material);
            clothingMesh.position.set(position.x, position.y, position.z);

            // For shoes, create a pair (left and right)
            if (category === 'shoes') {
                const clothingGroup = new THREE.Group();

                const leftShoe = clothingMesh.clone();
                leftShoe.position.set(-0.15, position.y, position.z);

                const rightShoe = clothingMesh.clone();
                rightShoe.position.set(0.15, position.y, position.z);

                clothingGroup.add(leftShoe);
                clothingGroup.add(rightShoe);

                // Add to scene
                this.scene.add(clothingGroup);

                // Store reference
                this.currentClothing[category] = clothingGroup;

                // Store metadata
                clothingGroup.userData = {
                    itemId: item.id,
                    type: category,
                    label: item.label || category
                };
            } else {
                // Add to scene
                this.scene.add(clothingMesh);

                // Store reference
                this.currentClothing[category] = clothingMesh;

                // Store metadata
                clothingMesh.userData = {
                    itemId: item.id,
                    type: category,
                    label: item.label || category
                };
            }

            // Update avatar preview if the function exists
            if (typeof window.updateAvatarPreview === 'function') {
                window.updateAvatarPreview();
            }

            this.showMessage(`Applied basic ${category} (fallback)`, 'warning');
            return true;

        } catch (error) {
            console.error('[AvatarClothingManager] Error creating fallback clothing:', error);
            this.showMessage(`Failed to apply ${category}`, 'error');
            return false;
        }
    }

    // Apply texture to model based on item color
    applyTextureToModel(model, item) {
        try {
            const color = this.getItemColor(item);

            if (!color) {
                return; // No color to apply
            }

            model.traverse((child) => {
                if (child.isMesh) {
                    // For simple color replacement
                    if (child.material) {
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
                }
            });

        } catch (error) {
            console.error('[AvatarClothingManager] Error applying texture:', error);
        }
    }

    // Get color from item
    getItemColor(item) {
        try {
            // Try different formats of color information
            if (item.color) {
                if (typeof item.color === 'string') {
                    return new THREE.Color(item.color);
                } else if (item.color.rgb) {
                    return new THREE.Color(
                        item.color.rgb[0] / 255,
                        item.color.rgb[1] / 255,
                        item.color.rgb[2] / 255
                    );
                }
            }

            // Default colors based on category
            const categoryColors = {
                tops: 0x3498db,    // Blue
                bottoms: 0x2c3e50, // Dark blue
                dresses: 0xe74c3c, // Red
                outerwear: 0x7f8c8d, // Gray
                shoes: 0x34495e,   // Dark gray
                accessories: 0xf1c40f // Yellow
            };

            const category = this.getCategory(item.type, item.label);
            return new THREE.Color(categoryColors[category] || 0xbdc3c7);

        } catch (error) {
            console.error('[AvatarClothingManager] Error getting item color:', error);
            return new THREE.Color(0xbdc3c7); // Silver fallback
        }
    }

    // Position clothing on avatar
    positionClothingOnAvatar(clothingModel, category) {
        // Scale and position adjustments
        clothingModel.scale.set(0.8, 0.8, 0.8);

        // Set position based on category
        switch (category) {
            case 'tops':
                clothingModel.position.set(0, 0.8, 0);
                break;
            case 'bottoms':
                clothingModel.position.set(0, 0.3, 0);
                break;
            case 'dresses':
                clothingModel.position.set(0, 0.5, 0);
                break;
            case 'outerwear':
                clothingModel.position.set(0, 0.8, 0);
                clothingModel.scale.set(0.9, 0.9, 0.9); // Slightly larger
                break;
            case 'shoes':
                clothingModel.position.set(0, 0.05, 0);
                break;
            case 'accessories':
                clothingModel.position.set(0.3, 1.0, 0.2);
                break;
            default:
                clothingModel.position.set(0, 0.6, 0);
        }
    }

    // Get model path for a category
    getModelPath(category) {
        return this.clothingPaths[category] || this.clothingPaths.tops;
    }

    // Get category from item type/label
    getCategory(itemType, itemLabel) {
        // Map Fashion-MNIST categories to our categories
        const categoryMapping = {
            't-shirt/top': 'tops',
            'shirt': 'tops',
            'pullover': 'tops',
            'trouser': 'bottoms',
            'dress': 'dresses',
            'coat': 'outerwear',
            'sandal': 'shoes',
            'sneaker': 'shoes',
            'ankle boot': 'shoes',
            'bag': 'accessories'
        };

        // Try to match by item type
        const lowerType = itemType.toLowerCase();
        if (categoryMapping[lowerType]) {
            return categoryMapping[lowerType];
        }

        // Direct category matches
        if (Object.values(categoryMapping).includes(lowerType)) {
            return lowerType;
        }

        // Try to match by label
        if (itemLabel) {
            const lowerLabel = itemLabel.toLowerCase();
            if (categoryMapping[lowerLabel]) {
                return categoryMapping[lowerLabel];
            }
        }

        // Default to tops if no match
        return 'tops';
    }

    // Remove clothing by category
    removeClothing(category) {
        if (this.currentClothing[category]) {
            if (this.debug) {
                console.log(`[AvatarClothingManager] Removing clothing: ${category}`);
            }

            const clothingItem = this.currentClothing[category];

            // Remove from scene
            this.scene.remove(clothingItem);

            // Dispose resources
            clothingItem.traverse((child) => {
                if (child.isMesh) {
                    if (child.geometry) {
                        child.geometry.dispose();
                    }

                    if (child.material) {
                        if (Array.isArray(child.material)) {
                            child.material.forEach(material => {
                                if (material.map) material.map.dispose();
                                material.dispose();
                            });
                        } else {
                            if (child.material.map) child.material.map.dispose();
                            child.material.dispose();
                        }
                    }
                }
            });

            // Delete reference
            delete this.currentClothing[category];

            // Update avatar preview
            if (typeof window.updateAvatarPreview === 'function') {
                window.updateAvatarPreview();
            }

            return true;
        }

        return false;
    }

    // Remove clothing by item ID
    removeClothingById(itemId) {
        for (const [category, clothingItem] of Object.entries(this.currentClothing)) {
            if (clothingItem.userData && clothingItem.userData.itemId === itemId) {
                return this.removeClothing(category);
            }
        }

        return false;
    }

    // Clear all clothing
    clearAllClothing() {
        for (const category in this.currentClothing) {
            this.removeClothing(category);
        }

        // Update avatar preview
        if (typeof window.updateAvatarPreview === 'function') {
            window.updateAvatarPreview();
        }

        return true;
    }

    // Show a message to the user
    showMessage(message, type = 'info') {
        // Use existing message function if available
        if (typeof window.showMessage === 'function') {
            window.showMessage(message, type);
            return;
        }

        // Create our own message display
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        messageElement.textContent = message;
        document.body.appendChild(messageElement);

        setTimeout(() => {
            messageElement.remove();
        }, 3000);
    }
}

// Initialize when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('[AvatarClothingManager] Document loaded, initializing...');

    // Wait for THREE.js and other dependencies to be available
    const checkDependencies = setInterval(() => {
        if (typeof THREE !== 'undefined' &&
            typeof THREE.GLTFLoader !== 'undefined' &&
            typeof ClothingMapper !== 'undefined') {

            clearInterval(checkDependencies);
            console.log('[AvatarClothingManager] Dependencies loaded, creating manager instance');

            // Create global instance
            window.avatarClothingManager = new AvatarClothingManager({
                debug: true,
                clothingMapper: new ClothingMapper()
            });

            // Set up event handlers for clothing items
            setupClothingItemEventListeners();
        }
    }, 100);
});

// Setup event listeners for all clothing items
function setupClothingItemEventListeners() {
    console.log('[AvatarClothingManager] Setting up clothing item event listeners');

    // Function to update all clothing item event listeners
    function updateClothingItemListeners() {
        // Get all wardrobe items
        const wardrobeItems = document.querySelectorAll('.wardrobe-item');

        wardrobeItems.forEach(item => {
            // Only add event listener if not already present
            if (!item.dataset.hasClothingListener) {
                item.addEventListener('click', async function(e) {
                    e.preventDefault();

                    // Toggle selection state
                    this.classList.toggle('selected');

                    // Only proceed if selected
                    if (this.classList.contains('selected')) {
                        // Get active tab to determine which manager to use
                        const isCustomTab = document.querySelector('#custom-avatar-section.active') !== null;

                        if (isCustomTab && window.avatarClothingManager) {
                            console.log('[Event] Custom tab clothing item clicked:', this.dataset.itemId);
                            await window.avatarClothingManager.handleClothingItemClick(this);
                        } else if (window.rpmManager) {
                            console.log('[Event] RPM tab clothing item clicked:', this.dataset.itemId);
                            // Let RPM manager handle it
                        }
                    } else {
                        // Handle deselection - remove the clothing
                        const isCustomTab = document.querySelector('#custom-avatar-section.active') !== null;
                        const itemId = this.dataset.itemId;

                        if (isCustomTab && window.avatarClothingManager && itemId) {
                            window.avatarClothingManager.removeClothingById(itemId);
                        } else if (window.rpmManager && window.rpmManager.removeClothing && itemId) {
                            window.rpmManager.removeClothing(itemId);
                        }
                    }
                });

                // Mark as having listener
                item.dataset.hasClothingListener = 'true';
            }
        });
    }

    // Update listeners now
    updateClothingItemListeners();

    // Set up MutationObserver to watch for new items being added
    const wardrobeContainer = document.querySelector('.avatar-page-content');
    if (wardrobeContainer) {
        const observer = new MutationObserver(mutations => {
            let hasNewItems = false;

            mutations.forEach(mutation => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    hasNewItems = true;
                }
            });

            if (hasNewItems) {
                updateClothingItemListeners();
            }
        });

        observer.observe(wardrobeContainer, {
            childList: true,
            subtree: true
        });
    }

    // Add event listener for tab switching to update avatar references
    const tabButtons = document.querySelectorAll('.avatar-tab');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.dataset.tab;

            if (tabName === 'custom' && window.avatarClothingManager) {
                // Update references when switching to custom tab
                if (window.avatar && window.scene) {
                    window.avatarClothingManager.setReferences(window.avatar, window.scene);
                    console.log('[AvatarClothingManager] Updated references after tab switch to custom');
                }
            }
        });
    });

    // Add event listener for the clear outfit buttons
    const clearOutfitCustomBtn = document.getElementById('clear-outfit-custom-btn');
    if (clearOutfitCustomBtn && window.avatarClothingManager) {
        clearOutfitCustomBtn.addEventListener('click', function() {
            window.avatarClothingManager.clearAllClothing();

            // Deselect all items in custom section
            document.querySelectorAll('#custom-avatar-section .wardrobe-item.selected').forEach(item => {
                item.classList.remove('selected');
            });

            if (typeof window.showMessage === 'function') {
                window.showMessage('All clothing removed', 'info');
            }
        });
    }
}