// ClothingRenderer.js - Improved implementation
class ClothingRenderer {
    constructor(avatarManager) {
        this.avatarManager = avatarManager;
        this.clothingItems = new Map(); // Store active clothing items
        this.textureLoader = new THREE.TextureLoader();
        this.gltfLoader = new THREE.GLTFLoader();

        // Initialize when the avatar is loaded
        if (this.avatarManager && this.avatarManager.avatarModel) {
            console.log('Avatar model available on initialization');
        } else {
            console.log('Waiting for avatar model to load...');
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
        if (!this.avatarManager || !this.avatarManager.scene) {
            console.error('Avatar scene not available');
            return false;
        }

        try {
            // Get clothing category
            const category = this.getCategory(clothingItem);
            console.log(`Rendering clothing item: ${clothingItem.label} (${category})`);

            // Remove any existing clothing in the same category
            this.removeClothingByCategory(category);

            // For now, create a simple placeholder mesh with the clothing texture
            const mesh = await this.createClothingMesh(clothingItem);
            if (!mesh) {
                console.error('Failed to create clothing mesh');
                return false;
            }

            // Add to scene
            this.avatarManager.scene.add(mesh);

            // Store reference
            this.clothingItems.set(category, {
                mesh: mesh,
                itemId: clothingItem._id
            });

            // Request a render update
            if (this.avatarManager.render) {
                this.avatarManager.render();
            }

            return true;
        } catch (error) {
            console.error('Error rendering clothing:', error);
            return false;
        }
    }

    // Create a simple clothing mesh with the item texture
    async createClothingMesh(clothingItem) {
        try {
            // Check if we have a file path
            if (!clothingItem.file_path) {
                console.error('No file path for clothing item');
                return null;
            }

            // Load texture from the file path
            const texture = await this.loadTexture(clothingItem.file_path);

            // Create geometry based on clothing type
            let geometry;
            const category = this.getCategory(clothingItem);

            switch(category) {
                case 'tops':
                    geometry = new THREE.PlaneGeometry(0.5, 0.5);
                    break;
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
}

// Make available globally
window.ClothingRenderer = ClothingRenderer;