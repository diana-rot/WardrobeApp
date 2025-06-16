
class ClothingOBJRenderer {
    constructor() {
        console.log(' Initializing Fixed ClothingOBJRenderer...');

        this.scene = null;
        this.avatar = null;
        this.currentClothing = new Map();
        this.objLoader = null;
        this.mtlLoader = null;

        this.initializeLoaders();
        console.log(' Fixed ClothingOBJRenderer initialized');
    }

    initializeLoaders() {
        try {
            if (typeof THREE === 'undefined') {
                throw new Error('THREE.js not loaded');
            }

            this.objLoader = new THREE.OBJLoader();

            if (THREE.MTLLoader) {
                this.mtlLoader = new THREE.MTLLoader();
            }

            console.log(' OBJ/MTL loaders initialized');
        } catch (error) {
            console.error(' Failed to initialize loaders:', error);
        }
    }

    setReferences(scene, avatar) {
        console.log(' Setting OBJ renderer references...');
        this.scene = scene;
        this.avatar = avatar;

        if (this.scene && this.avatar) {
            console.log(' OBJ renderer references set successfully');
            return true;
        } else {
            console.warn('️ Invalid references provided to OBJ renderer');
            return false;
        }
    }

    // Load clothing from database with improved fallback logic
    async loadClothingFromDatabase(itemId) {
        console.log(` Loading clothing from database: ${itemId}`);

        if (!this.scene || !this.avatar) {
            console.error(' Scene or avatar not set');
            throw new Error('Scene or avatar not set in OBJ renderer');
        }

        try {
            // Fetch item data using existing endpoint
            const response = await fetch(`/api/wardrobe/item/${itemId}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const itemData = await response.json();
            console.log(' Item data received:', itemData);

            if (!itemData.success) {
                throw new Error(itemData.error || 'Failed to fetch item data');
            }

            // PRIORITY 1: Check if item already has a saved 3D model path
            if (itemData.has_3d_model && itemData.model_3d_path) {
                console.log(` Found saved 3D model: ${itemData.model_3d_path}`);

                try {
                    const checkResponse = await fetch(itemData.model_3d_path, { method: 'HEAD' });
                    if (checkResponse.ok) {
                        console.log(` Loading saved OBJ file: ${itemData.model_3d_path}`);
                        return await this.loadOBJFile(itemData.model_3d_path, itemData);
                    }
                } catch (e) {
                    console.warn(` Saved model not accessible: ${e.message}`);
                }
            }

            // PRIORITY 2: Try to find OBJ files by model_task_id
            const userId = itemData.userId || itemData.user_id;
            const modelTaskId = itemData.model_task_id || itemData.modelTaskId;

            let objPath = null;

            // Look for OBJ files with the model_task_id
            if (modelTaskId && !modelTaskId.startsWith('auto_') && !modelTaskId.startsWith('fallback_')) {
                console.log(` Searching for OBJ files with model_task_id: ${modelTaskId}`);

                const patterns = [
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_0.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_1.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_2.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_3.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_4.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}.obj`
                ];

                for (const pattern of patterns) {
                    try {
                        const checkResponse = await fetch(pattern, { method: 'HEAD' });
                        if (checkResponse.ok) {
                            objPath = pattern;
                            console.log(` Found OBJ file: ${objPath}`);
                            break;
                        }
                    } catch (e) {
                        continue;
                    }
                }
            }

            // PRIORITY 3: Try to use the texture as a plane if available
            if (!objPath && (itemData.file_path || itemData.texture_preview_path)) {
                console.log(`No OBJ found, creating textured plane from image...`);
                return await this.createTexturedPlaneClothing(itemData);
            }

            // PRIORITY 4: Last resort - colored fallback
            if (!objPath) {
                console.warn(` No OBJ file found, creating colored fallback...`);
                return await this.createFallbackClothing(itemData);
            }

            // Load the found OBJ file
            console.log(` Loading OBJ file: ${objPath}`);
            return await this.loadOBJFile(objPath, itemData);

        } catch (error) {
            console.error(' Database loading failed:', error);
            console.log('Creating textured plane fallback...');

            const fallbackData = {
                _id: itemId,
                type: itemData?.type || 'tops',
                label: itemData?.label || 'Clothing Item',
                model_task_id: `fallback_${itemId}`,
                file_path: itemData?.file_path,
                color: itemData?.color
            };

            // Try textured plane first, then colored fallback
            if (itemData?.file_path || itemData?.texture_preview_path) {
                return await this.createTexturedPlaneClothing(fallbackData);
            } else {
                return await this.createFallbackClothing(fallbackData);
            }
        }
    }

    // NEW: Create textured plane clothing from image
    async createTexturedPlaneClothing(itemData) {
        console.log(' Creating textured plane clothing from image...');

        if (!this.scene || !this.avatar) {
            console.error(' Scene or avatar not set for textured plane');
            return false;
        }

        try {
            const clothingType = this.determineClothingType(itemData);
            const texturePath = itemData.file_path || itemData.texture_preview_path;

            if (!texturePath) {
                throw new Error('No texture path available');
            }

            // Create appropriate geometry for textured plane
            const geometry = this.createPlaneGeometry(clothingType);

            // Load texture and create material
            const material = await this.createTexturedPlaneMaterial(texturePath, itemData.color);

            const mesh = new THREE.Mesh(geometry, material);

            // Apply rotation
            this.applyClothingRotation(mesh, itemData);

            // Position on avatar
            this.positionClothingOnAvatar(mesh, clothingType);

            this.scene.add(mesh);

            const clothingData = {
                mesh: mesh,
                itemData: itemData,
                type: clothingType,
                category: itemData.type || clothingType,
                source: 'textured_plane'
            };

            this.currentClothing.set(itemData._id, clothingData);

            console.log(` Textured plane ${clothingType} created successfully`);
            return true;

        } catch (error) {
            console.error(' Textured plane creation failed:', error);
            return await this.createFallbackClothing(itemData);
        }
    }

    // Create plane geometry for different clothing types
    createPlaneGeometry(clothingType) {
        switch (clothingType) {
            case 'top':
                return new THREE.PlaneGeometry(1.2, 1.0);
            case 'bottom':
                return new THREE.PlaneGeometry(1.0, 1.2);
            case 'dress':
                return new THREE.PlaneGeometry(1.2, 1.8);
            case 'outerwear':
                return new THREE.PlaneGeometry(1.3, 1.1);
            case 'shoes':
                return new THREE.PlaneGeometry(0.6, 0.8);
            default:
                return new THREE.PlaneGeometry(1.0, 1.0);
        }
    }

    // Create material for textured plane
    async createTexturedPlaneMaterial(texturePath, colorData) {
        return new Promise((resolve) => {
            const loader = new THREE.TextureLoader();
            loader.load(
                texturePath,
                (texture) => {
                    console.log(' Texture loaded for plane:', texturePath);

                    // Extract color for tinting
                    let materialColor = 0xffffff;
                    if (colorData) {
                        if (typeof colorData === 'object' && colorData.rgb) {
                            const [r, g, b] = colorData.rgb;
                            materialColor = new THREE.Color(r / 255, g / 255, b / 255);
                        } else if (typeof colorData === 'string' && colorData.includes(' ')) {
                            const colorValues = colorData.split(' ').map(v => parseInt(v.trim()));
                            if (colorValues.length >= 3) {
                                const [r, g, b] = colorValues;
                                materialColor = new THREE.Color(r / 255, g / 255, b / 255);
                            }
                        }
                    }

                    const material = new THREE.MeshLambertMaterial({
                        map: texture,
                        color: materialColor,
                        transparent: true,
                        opacity: 0.95,
                        side: THREE.DoubleSide // Show both sides of the plane
                    });

                    resolve(material);
                },
                undefined,
                (error) => {
                    console.warn(' Texture loading failed for plane:', error);
                    resolve(new THREE.MeshLambertMaterial({
                        color: 0x808080,
                        transparent: false,
                        opacity: 1.0
                    }));
                }
            );
        });
    }

    // Load OBJ file with color and rotation
    async loadOBJFile(objPath, itemData) {
        return new Promise((resolve, reject) => {
            console.log(` Loading OBJ file: ${objPath}`);

            this.objLoader.load(
                objPath,
                (object) => {
                    console.log(' OBJ loaded, applying color and rotation...');
                    this.processLoadedObjectWithColor(object, itemData);
                    resolve(true);
                },
                (progress) => {
                    if (progress.lengthComputable) {
                        const percent = (progress.loaded / progress.total * 100).toFixed(1);
                        console.log(` Loading progress: ${percent}%`);
                    }
                },
                (error) => {
                    console.error(' OBJ loading failed:', error);
                    reject(error);
                }
            );
        });
    }

    // Process loaded OBJ with existing color format support
    processLoadedObjectWithColor(object, itemData) {
        console.log(' Processing OBJ with existing color format...');

        // Apply color using your existing format
        this.applyExistingColorFormat(object, itemData);

        // Apply rotation based on clothing type
        this.applyClothingRotation(object, itemData);

        // Position on avatar
        const clothingType = this.determineClothingType(itemData);
        this.positionClothingOnAvatar(object, clothingType);

        // Add to scene
        this.scene.add(object);

        // Store clothing data
        const clothingData = {
            mesh: object,
            itemData: itemData,
            type: clothingType,
            category: itemData.type || clothingType,
            source: 'real_obj_colored'
        };

        this.currentClothing.set(itemData._id, clothingData);

    }

    // Apply color using your EXISTING color format
    applyExistingColorFormat(object, itemData) {


        // Extract color using YOUR existing format
        let materialColor = 0x808080; // Default gray

        if (itemData.color) {


            if (typeof itemData.color === 'object') {
                if (itemData.color.rgb && Array.isArray(itemData.color.rgb)) {
                    const [r, g, b] = itemData.color.rgb;
                    materialColor = new THREE.Color(r / 255, g / 255, b / 255);

                }
            }
            else if (typeof itemData.color === 'string') {
                if (itemData.color.includes(' ')) {
                    const colorValues = itemData.color.split(' ').map(v => parseInt(v.trim()));
                    if (colorValues.length >= 3) {
                        const [r, g, b] = colorValues;
                        materialColor = new THREE.Color(r / 255, g / 255, b / 255);

                    }
                } else if (itemData.color.startsWith('#')) {
                    materialColor = new THREE.Color(itemData.color);

                }
            }
        }

        // Apply texture if available, with color tinting
        if (itemData.file_path || itemData.texture_preview_path) {
            const texturePath = itemData.file_path || itemData.texture_preview_path;

            const loader = new THREE.TextureLoader();
            loader.load(
                texturePath,
                (texture) => {
                    console.log(' Texture loaded, applying with color tint');

                    object.traverse((child) => {
                        if (child.isMesh) {
                            child.material = new THREE.MeshLambertMaterial({
                                map: texture,
                                color: materialColor,
                                transparent: false,
                                opacity: 1.0
                            });
                        }
                    });
                },
                undefined,
                (error) => {
                    console.warn(' Texture loading failed, using color only:', error);
                    this.applyColorOnlyToObject(object, materialColor);
                }
            );
        } else {
            this.applyColorOnlyToObject(object, materialColor);
        }
    }

    // Apply color only to object
    applyColorOnlyToObject(object, color) {
        console.log(' Applying color-only material');

        object.traverse((child) => {
            if (child.isMesh) {
                child.material = new THREE.MeshLambertMaterial({
                    color: color,
                    transparent: false,
                    opacity: 1.0
                });
            }
        });
    }

    // FIXED: Apply rotation based on clothing type - SINGLE FUNCTION, NO DUPLICATES
// REPLACE the applyClothingRotation function in your clothing-obj-renderer.js with this FORCED version:
// REPLACE the applyClothingRotation function with this version:
applyClothingRotation(object, itemData) {
    const clothingType = this.determineClothingType(itemData);
    console.log(` Applying CORRECTIVE rotation for ${clothingType}...`);

    // Reset all rotations first
    object.rotation.set(0, 0, 0);

    // Apply corrective rotations based on common export issues
    switch (clothingType) {
 case 'top':
            object.rotation.set(-Math.PI / 2, 0, Math.PI/2); // X=-90°, Y=180°, Z=0°
            break;
        case 'bottom':
            object.rotation.set(-Math.PI / 2, 0, Math.PI/2); // X=-90°, Y=180°, Z=0°
            break;
        case 'dress':
           object.rotation.set(-Math.PI / 2, 0, Math.PI/2); // X=-90°, Y=180°, Z=0°
            break;
        case 'outerwear':
            object.rotation.set(-Math.PI / 2, 0, Math.PI/2); // X=-90°, Y=180°, Z=0°
            break;
        case 'shoes':
             object.rotation.set(-Math.PI / 2, 0, Math.PI/2); // Only Y=180° for shoes
            break;
        default:
             object.rotation.set(-Math.PI / 2, 0, Math.PI/2);
    }

    console.log(`CORRECTIVE rotation applied: x=${object.rotation.x.toFixed(2)}, y=${object.rotation.y.toFixed(2)}, z=${object.rotation.z.toFixed(2)}`);
}
// FORCED rotation - applies rotation AFTER positioning to ensure it sticks


    // Create fallback clothing with existing color format
    async createFallbackClothing(itemData) {


        if (!this.scene || !this.avatar) {
            console.error(' Scene or avatar not set for fallback');
            return false;
        }

        try {
            const clothingType = this.determineClothingType(itemData);

            // Use plane geometry instead of 3D shapes for better appearance
            const geometry = this.createPlaneGeometry(clothingType);

            let material;
            if (itemData.file_path || itemData.texture_preview_path) {
                const texturePath = itemData.file_path || itemData.texture_preview_path;
                material = await this.createTexturedPlaneMaterial(texturePath, itemData.color);
            } else {
                material = this.createColoredMaterialWithExistingFormat(clothingType, itemData.color);
            }

            const mesh = new THREE.Mesh(geometry, material);

            this.applyClothingRotation(mesh, itemData);
            this.positionClothingOnAvatar(mesh, clothingType);

            this.scene.add(mesh);

            const clothingData = {
                mesh: mesh,
                itemData: itemData,
                type: clothingType,
                category: itemData.type || clothingType,
                source: 'improved_fallback'
            };

            this.currentClothing.set(itemData._id, clothingData);


            return true;

        } catch (error) {

            return false;
        }
    }


    createColoredMaterialWithExistingFormat(clothingType, colorData) {
        let materialColor = this.getDefaultColorForType(clothingType);

        if (colorData) {
            if (typeof colorData === 'object' && colorData.rgb) {
                const [r, g, b] = colorData.rgb;
                materialColor = new THREE.Color(r / 255, g / 255, b / 255);

            } else if (typeof colorData === 'string' && colorData.includes(' ')) {
                const colorValues = colorData.split(' ').map(v => parseInt(v.trim()));
                if (colorValues.length >= 3) {
                    const [r, g, b] = colorValues;
                    materialColor = new THREE.Color(r / 255, g / 255, b / 255);

                }
            }
        }

        return new THREE.MeshLambertMaterial({
            color: materialColor,
            transparent: false,
            opacity: 1.0,
            side: THREE.DoubleSide
        });
    }

    // Get default colors for different clothing types
    getDefaultColorForType(clothingType) {
        const defaultColors = {
            'top': 0x4CAF50,
            'bottom': 0x2196F3,
            'dress': 0xE91E63,
            'outerwear': 0x795548,
            'shoes': 0x424242,
            'default': 0x9E9E9E
        };

        return defaultColors[clothingType] || defaultColors.default;
    }

    // Determine clothing type
    determineClothingType(itemData) {
        const label = (itemData.label || itemData.type || '').toLowerCase();
        const category = (itemData.category || itemData.type || '').toLowerCase();

        if (label.includes('shirt') || label.includes('top') || label.includes('pullover') || category === 'tops') {
            return 'top';
        } else if (label.includes('trouser') || label.includes('pant') || category === 'bottoms') {
            return 'bottom';
        } else if (label.includes('dress') || category === 'dresses') {
            return 'dress';
        } else if (label.includes('coat') || label.includes('jacket') || category === 'outerwear') {
            return 'outerwear';
        } else if (label.includes('shoe') || label.includes('sandal') || label.includes('boot') || category === 'shoes') {
            return 'shoes';
        } else {
            return 'top';
        }
    }

    // Position clothing on avatar with better scaling
    positionClothingOnAvatar(mesh, clothingType) {
        if (!this.avatar) {
            console.warn(' No avatar available for positioning');
            return;
        }

        const avatarBox = new THREE.Box3().setFromObject(this.avatar);
        const avatarHeight = avatarBox.max.y - avatarBox.min.y;
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());

        switch (clothingType) {
            case 'top':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y + avatarHeight * 0.15,
                    avatarCenter.z + 0.1
                );
                break;
            case 'bottom':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y - avatarHeight * 0.15,
                    avatarCenter.z + 0.1
                );
                break;
            case 'dress':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y - avatarHeight * 0.05,
                    avatarCenter.z + 0.1
                );
                break;
            case 'outerwear':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y + avatarHeight * 0.15,
                    avatarCenter.z + 0.15
                );
                break;
            case 'shoes':
                mesh.position.set(
                    avatarCenter.x,
                    avatarBox.min.y + 0.05,
                    avatarCenter.z + 0.05
                );
                break;
            default:
                mesh.position.copy(avatarCenter);
                mesh.position.z += 0.1;
        }

        // Better scaling for different clothing types
        let scale;
        switch (clothingType) {
            case 'dress':
                scale = avatarHeight / 2.2;
                break;
            case 'top':
                scale = avatarHeight / 2.8;
                break;
            case 'bottom':
                scale = avatarHeight / 3.0;
                break;
            default:
                scale = avatarHeight / 2.5;
        }

        mesh.scale.setScalar(scale);

    }

    // Remove clothing
    async removeClothing(itemId) {


        if (!this.currentClothing.has(itemId)) {
            console.warn(` Clothing ${itemId} not found`);
            return false;
        }

        try {
            const clothingData = this.currentClothing.get(itemId);

            if (clothingData.mesh && this.scene) {
                this.scene.remove(clothingData.mesh);

                if (clothingData.mesh.geometry) {
                    clothingData.mesh.geometry.dispose();
                }

                if (clothingData.mesh.material) {
                    if (Array.isArray(clothingData.mesh.material)) {
                        clothingData.mesh.material.forEach(mat => mat.dispose());
                    } else {
                        clothingData.mesh.material.dispose();
                    }
                }
            }

            this.currentClothing.delete(itemId);
            console.log(` Clothing ${itemId} removed successfully`);
            return true;

        } catch (error) {
            console.error('Error removing clothing:', error);
            return false;
        }
    }

    // Clear all clothing
    async clearAllClothing() {
        console.log(' Clearing all clothing...');

        const itemIds = Array.from(this.currentClothing.keys());
        let removedCount = 0;

        for (const itemId of itemIds) {
            const success = await this.removeClothing(itemId);
            if (success) {
                removedCount++;
            }
        }

        console.log(`Cleared ${removedCount} clothing items`);
        return removedCount;
    }

    // Get active clothing info
    getActiveClothing() {
        const activeItems = [];

        for (const [itemId, clothingData] of this.currentClothing.entries()) {
            activeItems.push({
                id: itemId,
                name: clothingData.itemData?.label || 'Unknown Item',
                type: clothingData.type,
                category: clothingData.category,
                source: clothingData.source
            });
        }

        return activeItems;
    }
}

// Initialize and make globally available
console.log(' Fixed ClothingOBJRenderer class defined (no duplicates)');

function initializeClothingOBJRenderer() {
    if (typeof THREE !== 'undefined' && THREE.OBJLoader) {
        window.clothingOBJRenderer = new ClothingOBJRenderer();
        console.log(' Fixed ClothingOBJRenderer instance created globally');
        return true;
    } else {
        console.log(' Waiting for THREE.js...');
        return false;
    }
}

if (!initializeClothingOBJRenderer()) {
    const initInterval = setInterval(() => {
        if (initializeClothingOBJRenderer()) {
            clearInterval(initInterval);
        }
    }, 100);

    setTimeout(() => {
        clearInterval(initInterval);
        if (!window.clothingOBJRenderer) {
            console.error(' Failed to initialize Fixed ClothingOBJRenderer after 10 seconds');
        }
    }, 10000);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ClothingOBJRenderer;
}