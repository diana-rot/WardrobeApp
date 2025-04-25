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
        this.camera.position.set(0, 1.6, 2.5);

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
        this.controls.target.set(0, 1.2, 0);
        
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

            // Remove existing avatar if any
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel.traverse((node) => {
                    if (node.isMesh) {
                        if (node.material) {
                            node.material.dispose();
                        }
                        if (node.geometry) {
                            node.geometry.dispose();
                        }
                    }
                });
                this.avatarModel = null;
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

        // Set initial camera position - moved even further back (z from 3.5 to 4.5)
        this.camera.position.set(0, 1.6, 4.5);
        this.controls.target.set(0, 1.2, 0);
        
        // Set camera limits - adjusted for further viewing
        this.controls.minDistance = 3.0;  // Increased minimum distance
        this.controls.maxDistance = 6.0;  // Increased maximum distance
        this.controls.minPolarAngle = Math.PI / 4;
        this.controls.maxPolarAngle = Math.PI / 2;
        
        // Update camera and controls
        this.camera.updateProjectionMatrix();
        this.controls.update();
    }

    // Add try-on related methods to RPMAvatarManager class
    tryOnWardrobeItem = async function(itemData) {
        if (!itemData || !itemData.file_path) {
            throw new Error('Invalid wardrobe item data');
        }

        try {
            this.showLoadingOverlay('Preparing item for try-on...');

            // Request 3D model generation from the server
            const response = await fetch('/api/generate-3d-clothing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    wardrobe_item_id: itemData._id,
                    category: itemData.label
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate 3D model');
            }

            const modelData = await response.json();

            // Load the generated 3D model
            if (modelData.model_url) {
                await this.loadClothingModel(modelData.model_url, itemData.label);
                this.showMessage('Item tried on successfully!', 'success');
            } else {
                throw new Error('No model URL received');
            }

        } catch (error) {
            console.error('Try-on error:', error);
            this.showMessage('Failed to try on item: ' + error.message, 'error');
        } finally {
            this.hideLoadingOverlay();
        }
    };

    loadClothingModel = async function(modelUrl, category) {
        const loader = new THREE.GLTFLoader();
        
        try {
            const gltf = await new Promise((resolve, reject) => {
                loader.load(modelUrl, resolve, undefined, reject);
            });

            // Remove existing clothing of the same category if any
            this.removeExistingClothing(category);

            // Add the new clothing model
            const clothingModel = gltf.scene;
            clothingModel.userData.category = category;

            // Scale and position the clothing
            this.fitClothingToAvatar(clothingModel);

            // Add to scene
            this.scene.add(clothingModel);
            
            // Store reference to current clothing
            if (!this.currentClothing) {
                this.currentClothing = {};
            }
            this.currentClothing[category] = clothingModel;

        } catch (error) {
            console.error('Error loading clothing model:', error);
            throw error;
        }
    };

    removeExistingClothing = function(category) {
        if (this.currentClothing && this.currentClothing[category]) {
            const existingClothing = this.currentClothing[category];
            this.scene.remove(existingClothing);
            existingClothing.traverse((node) => {
                if (node.isMesh) {
                    if (node.geometry) node.geometry.dispose();
                    if (node.material) {
                        if (Array.isArray(node.material)) {
                            node.material.forEach(mat => mat.dispose());
                        } else {
                            node.material.dispose();
                        }
                    }
                }
            });
            delete this.currentClothing[category];
        }
    };

    fitClothingToAvatar = function(clothingModel) {
        if (!this.avatarModel) {
            console.warn('No avatar model to fit clothing to');
            return;
        }

        // Get avatar dimensions
        const avatarBox = new THREE.Box3().setFromObject(this.avatarModel);
        const avatarSize = avatarBox.getSize(new THREE.Vector3());
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());

        // Get clothing dimensions
        const clothingBox = new THREE.Box3().setFromObject(clothingModel);
        const clothingSize = clothingBox.getSize(new THREE.Vector3());

        // Calculate scale to match avatar size
        const scale = avatarSize.y / clothingSize.y;
        clothingModel.scale.setScalar(scale);

        // Position clothing on avatar
        clothingModel.position.copy(avatarCenter);
    };
}

// Export the class
window.RPMAvatarManager = RPMAvatarManager; 