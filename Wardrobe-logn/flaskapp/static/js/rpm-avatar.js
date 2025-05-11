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

        // Try to load saved avatar on initialization
        this.loadSavedAvatar();
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
        this.camera.position.set(0, -2.0, 2.5);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.shadowMap.enabled = true;
        
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
        this.addGroundPlane();
        
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
    async loadClothing(itemId, imageUrl, itemType) {
        if (!this.avatarModel) {
            console.error('No avatar loaded. Please load an avatar first.');
            return false;
        }

        try {
            console.log('Loading clothing item:', { itemId, imageUrl, itemType });
            this.showLoadingOverlay('Loading clothing...');

            // Initialize GLTFLoader if not exists
            if (!this.gltfLoader) {
                this.gltfLoader = new THREE.GLTFLoader();
            }

            // Load the top.glb model
            const modelPath = '/static/models/clothing/top.glb';
            console.log('Loading model from:', modelPath);

            try {
                const gltf = await new Promise((resolve, reject) => {
                    this.gltfLoader.load(
                        modelPath,
                        resolve,
                        (xhr) => {
                            const percent = Math.round((xhr.loaded / xhr.total) * 100);
                            console.log(`Loading model: ${percent}%`);
                        },
                        reject
                    );
                });

                console.log('Model loaded successfully:', gltf);

                // Remove existing clothing if any
                if (this.currentClothing) {
                    this.scene.remove(this.currentClothing);
                    this.currentClothing = null;
                }

                const model = gltf.scene;

                // Make sure model and its children are visible and castShadow
                model.traverse((node) => {
                    if (node.isMesh) {
                        node.castShadow = true;
                        node.receiveShadow = true;
                        node.visible = true;
                        
                        // Ensure material is properly configured
                        if (node.material) {
                            node.material.transparent = true;
                            node.material.needsUpdate = true;
                            node.material.side = THREE.DoubleSide;
                        }
                    }
                });

                // Calculate bounding box to help with positioning
                const bbox = new THREE.Box3().setFromObject(model);
                const size = new THREE.Vector3();
                bbox.getSize(size);
                const center = new THREE.Vector3();
                bbox.getCenter(center);

                // Position relative to avatar's chest area
                model.position.set(0, 1.4, 0); // Y position at chest level
                model.scale.set(0.5, 0.5, 0.5); // Start with half scale, adjust as needed

                // Add to avatar if loaded, otherwise to scene
                if (this.avatarModel) {
                    this.avatarModel.add(model);
                    console.log('[DEBUG] top.glb added as child of avatarModel');
                } else {
                    this.scene.add(model);
                    console.warn('[DEBUG] avatarModel not present, added top.glb to scene');
                }
                this.currentClothing = model;

                // Store reference
                if (!this.clothingItems) {
                    this.clothingItems = new Map();
                }
                this.clothingItems.set(itemId, {
                    mesh: model,
                    type: itemType
                });

                // Log model position, scale, and parent
                console.log('[DEBUG] top.glb position:', model.position);
                console.log('[DEBUG] top.glb scale:', model.scale);
                console.log('[DEBUG] top.glb parent:', model.parent);

                // Add debug helpers
                if (this.debug) {
                    // Add bounding box helper
                    const boxHelper = new THREE.BoxHelper(model, 0xff0000);
                    this.scene.add(boxHelper);

                    // Add axes helper
                    const axesHelper = new THREE.AxesHelper(1);
                    model.add(axesHelper);
                }

                return true;
            } catch (error) {
                console.error('Error loading GLB model:', error);
                return false;
            }
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

    // Clear all clothing
    clearAllClothing() {
        if (!this.clothingItems) {
            return;
        }

        for (const [itemId, item] of this.clothingItems.entries()) {
            this.removeClothing(itemId);
        }

        this.clothingItems.clear();
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