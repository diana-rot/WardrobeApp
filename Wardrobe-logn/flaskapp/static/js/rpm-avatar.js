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
                
                // Update preview if it exists
                const preview = document.getElementById('rpm-avatar-preview');
                if (preview) {
                    preview.innerHTML = `<img src="${data.avatarUrl}" alt="RPM Avatar">`;
                }
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
    
    async loadAvatar(avatarUrl) {
        if (this.isLoading) {
            console.warn('Avatar is already loading');
            return;
        }

        try {
            this.isLoading = true;
            this.showLoadingOverlay('Loading avatar...');
            
            // Save the avatar URL first
            await this.saveAvatarUrl(avatarUrl);
            
            // Convert avatar URL to GLB if necessary
            const glbUrl = this.getGLBUrlFromAvatarUrl(avatarUrl);
            console.log('Loading avatar from URL:', glbUrl);
            
            // Load the avatar model
            const loader = new THREE.GLTFLoader();
            const gltf = await new Promise((resolve, reject) => {
                loader.load(
                    glbUrl,
                    resolve,
                    (progress) => {
                        const percent = (progress.loaded / progress.total * 100).toFixed(0);
                        console.log('Loading progress:', percent + '%');
                        this.updateLoadingProgress(percent);
                    },
                    reject
                );
            });
            
            // Remove existing avatar if any
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel.traverse((child) => {
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
            
            // Add new avatar to scene
            this.avatarModel = gltf.scene;
            this.avatarModel.scale.set(1, 1, 1);
            this.avatarModel.position.y = 0;
            
            // Enable shadows
            this.avatarModel.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            
            this.scene.add(this.avatarModel);
            
            // Save avatar URL
            this.avatarUrl = avatarUrl;
            
            // Center camera on avatar
            this.centerCameraOnAvatar();
            
            console.log('Avatar loaded successfully');
            this.onAvatarLoaded(this.avatarModel);
            
        } catch (error) {
            console.error('Error loading avatar:', error);
            this.onAvatarError(error);
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
        if (!this.avatarModel || !this.camera || !this.controls) return;
        
        const box = new THREE.Box3().setFromObject(this.avatarModel);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / Math.tan(fov / 2)) * 1.2;
        
        this.camera.position.set(
            center.x + cameraZ * 0.3,
            center.y + size.y / 2.5,
            center.z + cameraZ
        );
        
        this.controls.target.set(
            center.x,
            center.y + size.y / 2,
            center.z
        );
        
        this.camera.updateProjectionMatrix();
        this.controls.update();
    }
    
    showLoadingOverlay(message) {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }
    
    updateLoadingProgress(percent) {
        const progress = document.querySelector('.loading-progress');
        if (progress) {
            progress.textContent = `Loading: ${percent}%`;
        }
    }
    
    hideLoadingOverlay() {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
}

// Export the class
window.RPMAvatarManager = RPMAvatarManager; 