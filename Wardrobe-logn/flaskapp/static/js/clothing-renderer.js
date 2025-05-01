// ClothingRenderer.js - Improved implementation
class ClothingRenderer {
    constructor(avatarManager) {
        console.log('ClothingRenderer constructor called with:', {
            hasAvatarManager: !!avatarManager,
            hasScene: avatarManager ? !!avatarManager.scene : false,
            hasAvatarModel: avatarManager ? !!avatarManager.avatarModel : false
        });

        if (!avatarManager) {
            console.error('Avatar manager is required for ClothingRenderer');
            throw new Error('Avatar manager is required');
        }

        if (!avatarManager.scene) {
            console.error('Avatar manager scene is required for ClothingRenderer');
            throw new Error('Avatar manager scene is required');
        }

        // Debug scene setup
        console.log('Scene setup:', {
            children: avatarManager.scene.children.length,
            camera: !!avatarManager.camera,
            renderer: !!avatarManager.renderer,
            cameraPosition: avatarManager.camera ? avatarManager.camera.position.toArray() : null,
            rendererSize: avatarManager.renderer ? {
                width: avatarManager.renderer.domElement.width,
                height: avatarManager.renderer.domElement.height
            } : null
        });

        this.avatarManager = avatarManager;
        this.clothingItems = new Map(); // Store active clothing items
        this.isInitialized = false;
        
        // Initialize loaders
        if (typeof THREE === 'undefined') {
            console.error('THREE.js is not loaded');
            throw new Error('THREE.js is not loaded');
        }

        this.textureLoader = new THREE.TextureLoader();
        
        // Initialize GLTFLoader with proper error handling
        if (typeof THREE.GLTFLoader === 'undefined') {
            console.error('GLTFLoader is not loaded. Make sure to include the GLTFLoader script.');
            throw new Error('GLTFLoader is not loaded');
        }

        this.gltfLoader = new THREE.GLTFLoader();
        console.log('GLTFLoader initialized successfully');

        // Debug renderer setup
        if (this.avatarManager.renderer) {
            console.log('Renderer properties:', {
                shadowMap: this.avatarManager.renderer.shadowMap.enabled,
                shadowMapType: this.avatarManager.renderer.shadowMap.type,
                pixelRatio: this.avatarManager.renderer.getPixelRatio(),
                size: {
                    width: this.avatarManager.renderer.domElement.width,
                    height: this.avatarManager.renderer.domElement.height
                }
            });
        }

        // Map clothing categories
        this.categories = {
            'tops': ['T-shirt/top', 'Shirt', 'Pullover'],
            'bottoms': ['Trouser'],
            'dresses': ['Dress'],
            'outerwear': ['Coat'],
            'shoes': ['Sandal', 'Sneaker', 'Ankle boot'],
            'accessories': ['Bag']
        };

        // Initialize if avatar model is already available
        if (this.avatarManager.avatarModel) {
            console.log('Avatar model available on initialization');
            this.setupLighting();
            this.isInitialized = true;
        } else {
            console.log('Waiting for avatar model to load...');
        }

        console.log('ClothingRenderer initialization complete. Status:', {
            isInitialized: this.isInitialized,
            hasScene: !!this.avatarManager.scene,
            hasRenderer: !!this.avatarManager.renderer,
            hasAvatarModel: !!this.avatarManager.avatarModel
        });
    }

    // Check if the renderer is properly initialized
    isReady() {
        return this.isInitialized && 
               this.avatarManager && 
               this.avatarManager.scene && 
               this.avatarManager.avatarModel;
    }

    // Initialize the renderer when avatar is loaded
    initialize() {
        if (this.avatarManager && this.avatarManager.avatarModel) {
            console.log('Initializing ClothingRenderer with avatar model');
            this.setupLighting();
            this.isInitialized = true;
            return true;
        }
        return false;
    }

    setupLighting() {
        if (!this.avatarManager || !this.avatarManager.scene) return;

        console.log('Setting up lighting...');

        // Remove any existing lights
        this.avatarManager.scene.traverse((node) => {
            if (node.isLight) {
                console.log('Removing existing light:', node.type);
                this.avatarManager.scene.remove(node);
            }
        });

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.avatarManager.scene.add(ambientLight);
        console.log('Added ambient light');

        // Add directional light
        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(5, 5, 5);
        dirLight.castShadow = true;
        
        // Configure shadow properties
        dirLight.shadow.mapSize.width = 1024;
        dirLight.shadow.mapSize.height = 1024;
        dirLight.shadow.camera.near = 0.1;
        dirLight.shadow.camera.far = 10;
        
        this.avatarManager.scene.add(dirLight);
        console.log('Added directional light:', {
            position: dirLight.position.toArray(),
            intensity: dirLight.intensity,
            castShadow: dirLight.castShadow
        });

        // Add a helper to visualize the light
        const helper = new THREE.DirectionalLightHelper(dirLight, 1);
        this.avatarManager.scene.add(helper);
        console.log('Added light helper');
    }

    // Get category for a clothing item
    getCategory(clothingItem) {
        const label = clothingItem.label || '';

        for (const [category, labels] of Object.entries(this.categories)) {
            if (labels.includes(label)) {
                return category;
            }
        }

        return 'unknown';
    }

    // Render clothing on avatar
    async renderClothing(clothingItem) {
        console.log('renderClothing called with item:', clothingItem);
        if (!this.isReady()) {
            console.error('ClothingRenderer not properly initialized', {
                isInitialized: this.isInitialized,
                hasAvatarManager: !!this.avatarManager,
                hasScene: this.avatarManager ? !!this.avatarManager.scene : false,
                hasAvatarModel: this.avatarManager ? !!this.avatarManager.avatarModel : false
            });
            return false;
        }

        try {
            // Get clothing category
            const category = this.getCategory(clothingItem);
            console.log(`Rendering clothing item: ${clothingItem.label} (${category})`);

            // Remove any existing clothing in the same category
            this.removeClothingByCategory(category);

            // Create the clothing mesh
            console.log('Creating clothing mesh...');
            const mesh = await this.createClothingMesh(clothingItem);
            if (!mesh) {
                console.error('Failed to create clothing mesh');
                return false;
            }
            console.log('Clothing mesh created successfully');

            // For 3D models, we need to handle them differently than 2D planes
            if (category === 'tops') {
                console.log('Processing 3D model for top...');
                // Apply any necessary transformations for the 3D model
                mesh.traverse((node) => {
                    if (node.isMesh) {
                        node.castShadow = true;
                        node.receiveShadow = true;
                    }
                });
            }

            // Add to scene
            console.log('Adding mesh to scene...');
            this.avatarManager.scene.add(mesh);
            console.log('Mesh added to scene');

            // Store reference
            this.clothingItems.set(category, {
                mesh: mesh,
                itemId: clothingItem._id
            });

            // Request a render update
            if (this.avatarManager.render) {
                console.log('Requesting render update...');
                this.avatarManager.render();
            }

            return true;
        } catch (error) {
            console.error('Error rendering clothing:', error);
            console.error('Error details:', {
                message: error.message,
                stack: error.stack
            });
            return false;
        }
    }

    // Create a simple clothing mesh with the item texture
    async createClothingMesh(clothingItem) {
        try {
            const category = this.getCategory(clothingItem);
            console.log('Creating clothing mesh for category:', category, 'with item:', clothingItem);
            
            // Only load 3D GLB model for tops
            if (category === 'tops' && clothingItem.modelPath) {
                console.log('Loading 3D model for top category with path:', clothingItem.modelPath);
                console.log('GLTFLoader available:', !!this.gltfLoader);
                
                return new Promise((resolve, reject) => {
                    console.log('Starting GLB model load...');
                    this.gltfLoader.load(
                        clothingItem.modelPath,
                        (gltf) => {
                            console.log('GLB model loaded successfully');
                            console.log('Model structure:', this.logModelStructure(gltf.scene));
                            const model = gltf.scene;
                            
                            // Apply transformations
                            model.scale.set(1, 1, 1);
                            model.position.set(0, 1.4, 0);
                            model.rotation.set(0, Math.PI, 0);
                            
                            // Make sure materials are visible
                            model.traverse((node) => {
                                if (node.isMesh) {
                                    console.log('Processing mesh:', node.name || 'unnamed');
                                    node.visible = true;
                                    node.frustumCulled = false;
                                    
                                    if (node.material) {
                                        node.material.side = THREE.DoubleSide;
                                        node.material.transparent = true;
                                        node.material.opacity = 1;
                                        node.material.needsUpdate = true;
                                        console.log('Material properties:', {
                                            side: node.material.side,
                                            transparent: node.material.transparent,
                                            opacity: node.material.opacity
                                        });
                                    }
                                    
                                    node.castShadow = true;
                                    node.receiveShadow = true;
                                }
                            });
                            
                            resolve(model);
                        },
                        (progress) => {
                            const percent = (progress.loaded / progress.total * 100).toFixed(2);
                            console.log('Loading progress:', percent + '%');
                        },
                        (error) => {
                            console.error('Error loading GLB model:', error);
                            console.error('Error details:', {
                                message: error.message,
                                stack: error.stack
                            });
                            reject(error);
                        }
                    );
                });
            }

            // For non-tops or if no model path, create 2D plane with texture
            if (!clothingItem.textureUrl) {
                console.error('No texture URL for clothing item');
                return null;
            }

            // Load texture from the URL
            const texture = await this.loadTexture(clothingItem.textureUrl);

            // Create geometry based on category
            let geometry;
            switch(category) {
                case 'bottoms':
                    geometry = new THREE.PlaneGeometry(0.4, 0.6);
                    break;
                case 'dresses':
                    geometry = new THREE.PlaneGeometry(0.5, 0.8);
                    break;
                case 'outerwear':
                    geometry = new THREE.PlaneGeometry(0.6, 0.6);
                    break;
                case 'shoes':
                    geometry = new THREE.PlaneGeometry(0.3, 0.2);
                    break;
                default:
                    geometry = new THREE.PlaneGeometry(0.4, 0.4);
            }

            // Create material with the texture
            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                side: THREE.DoubleSide
            });

            // Create mesh
            const mesh = new THREE.Mesh(geometry, material);

            // Position based on category
            this.positionClothingMesh(mesh, category);

            return mesh;
        } catch (error) {
            console.error('Error creating clothing mesh:', error);
            return null;
        }
    }

    // Position the clothing mesh based on category
    positionClothingMesh(mesh, category) {
        switch(category) {
            case 'tops':
                mesh.position.set(0, 1.4, 0.1);
                break;
            case 'bottoms':
                mesh.position.set(0, 0.8, 0.1);
                break;
            case 'dresses':
                mesh.position.set(0, 1.1, 0.1);
                break;
            case 'outerwear':
                mesh.position.set(0, 1.3, 0.15);
                break;
            case 'shoes':
                mesh.position.set(0, 0.05, 0.2);
                break;
            case 'accessories':
                mesh.position.set(0.3, 1.0, 0.2);
                break;
            default:
                mesh.position.set(0, 1.0, 0.1);
        }
    }

    // Load texture from URL
    loadTexture(url) {
        return new Promise((resolve, reject) => {
            this.textureLoader.load(
                url,
                (texture) => resolve(texture),
                undefined,
                (error) => {
                    console.error('Error loading texture:', error);
                    reject(error);
                }
            );
        });
    }

    // Remove clothing by category
    removeClothingByCategory(category) {
        if (this.clothingItems.has(category)) {
            const { mesh } = this.clothingItems.get(category);

            if (mesh) {
                // Remove from scene
                this.avatarManager.scene.remove(mesh);

                // Dispose resources
                if (mesh.geometry) mesh.geometry.dispose();

                if (mesh.material) {
                    if (Array.isArray(mesh.material)) {
                        mesh.material.forEach(material => {
                            if (material.map) material.map.dispose();
                            material.dispose();
                        });
                    } else {
                        if (mesh.material.map) mesh.material.map.dispose();
                        mesh.material.dispose();
                    }
                }
            }

            // Remove from map
            this.clothingItems.delete(category);

            // Request a render update
            if (this.avatarManager.render) {
                this.avatarManager.render();
            }
        }
    }

    // Clear all clothing
    clearAllClothing() {
        for (const category of this.clothingItems.keys()) {
            this.removeClothingByCategory(category);
        }
    }

    // Helper method to count meshes in a model
    countMeshes(object) {
        let count = 0;
        object.traverse((node) => {
            if (node.isMesh) count++;
        });
        return count;
    }

    // Helper method to log model structure
    logModelStructure(object, level = 0) {
        let structure = '';
        const indent = '  '.repeat(level);
        
        structure += `${indent}${object.type}`;
        if (object.name) structure += ` (${object.name})`;
        if (object.isMesh) structure += ' [MESH]';
        structure += '\n';
        
        object.children.forEach(child => {
            structure += this.logModelStructure(child, level + 1);
        });
        
        return structure;
    }
}

// Make sure ClothingRenderer is available globally
if (typeof window !== 'undefined') {
    console.log('Making ClothingRenderer available globally...');
    window.ClothingRenderer = ClothingRenderer;
    console.log('ClothingRenderer is now available as:', typeof window.ClothingRenderer);
}