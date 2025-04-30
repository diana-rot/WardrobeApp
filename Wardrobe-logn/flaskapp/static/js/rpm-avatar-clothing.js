// RPMAvatarManager.js - Manages the Ready Player Me avatar
class RPMAvatarManager {
    constructor(options = {}) {
        // Get container element
        this.container = options.container || document.body;

        // Scene setup
        this.initScene();

        // Avatar model reference
        this.avatarModel = null;
        this.avatarUrl = null;

        // Animation mixer
        this.mixer = null;
        this.clock = new THREE.Clock();

        // Start render loop
        this.animate();

        // Handle window resize
        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    // Initialize Three.js scene
    initScene() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf5f5f5);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            45,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 1.5, 2);
        this.camera.lookAt(0, 1, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Add renderer to container
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 1, 0);
        this.controls.update();

        // Lights
        this.addLights();

        // Floor (optional)
        this.addFloor();
    }

    // Add lighting to scene
    addLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Directional light (sun)
        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(1, 2, 3);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 1024;
        dirLight.shadow.mapSize.height = 1024;
        this.scene.add(dirLight);

        // Additional light from the front
        const frontLight = new THREE.DirectionalLight(0xffffff, 0.8);
        frontLight.position.set(0, 1, 2);
        this.scene.add(frontLight);
    }

    // Add floor to scene
    addFloor() {
        const floorGeometry = new THREE.CircleGeometry(2, 32);
        const floorMaterial = new THREE.MeshStandardMaterial({
            color: 0xeeeeee,
            roughness: 0.8,
            metalness: 0.2
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        floor.position.y = 0;
        this.scene.add(floor);
    }

    // Load avatar from URL
    loadAvatar(url) {
        return new Promise((resolve, reject) => {
            // Clear any existing avatar
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel = null;
            }

            // Load new avatar
            const loader = new THREE.GLTFLoader();
            loader.load(
                url,
                (gltf) => {
                    // Add to scene
                    this.avatarModel = gltf.scene;
                    this.avatarModel.traverse((node) => {
                        if (node.isMesh) {
                            node.castShadow = true;
                        }
                    });

                    // Scale and position avatar
                    this.avatarModel.scale.set(1, 1, 1);
                    this.avatarModel.position.set(0, 0, 0);

                    // Add to scene
                    this.scene.add(this.avatarModel);

                    // Set up animations if available
                    if (gltf.animations && gltf.animations.length) {
                        this.mixer = new THREE.AnimationMixer(this.avatarModel);
                        const idleAction = this.mixer.clipAction(gltf.animations[0]);
                        idleAction.play();
                    }

                    // Store URL
                    this.avatarUrl = url;

                    console.log('Avatar loaded successfully');
                    resolve(this.avatarModel);
                },
                (xhr) => {
                    console.log(`Loading avatar: ${(xhr.loaded / xhr.total * 100).toFixed(2)}%`);
                },
                (error) => {
                    console.error('Error loading avatar:', error);
                    reject(error);
                }
            );
        });
    }

    // Load saved avatar from user profile
    async loadSavedAvatar() {
        try {
            const response = await fetch('/api/avatar/get-rpm-avatar');
            const data = await response.json();

            if (data.success && (data.avatarUrl || data.localPath)) {
                const url = data.localPath || data.avatarUrl;
                return await this.loadAvatar(url);
            } else {
                console.warn('No saved avatar found');
                return null;
            }
        } catch (error) {
            console.error('Error loading saved avatar:', error);
            return null;
        }
    }

    // Animation loop
    animate() {
        requestAnimationFrame(this.animate.bind(this));

        // Update animation mixer
        if (this.mixer) {
            this.mixer.update(this.clock.getDelta());
        }

        // Update controls
        if (this.controls) {
            this.controls.update();
        }

        this.render();
    }

    // Render the scene
    render() {
        this.renderer.render(this.scene, this.camera);
    }

    // Handle window resize
    onWindowResize() {
        if (!this.camera || !this.renderer || !this.container) return;

        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    // Dispose resources
    dispose() {
        // Stop animation loop
        cancelAnimationFrame(this.animationFrameId);

        // Remove event listeners
        window.removeEventListener('resize', this.onWindowResize);

        // Dispose controls
        if (this.controls) {
            this.controls.dispose();
        }

        // Remove DOM element
        if (this.container && this.renderer) {
            this.container.removeChild(this.renderer.domElement);
        }

        // Dispose renderer
        if (this.renderer) {
            this.renderer.dispose();
        }
    }
}

// Make available globally
window.RPMAvatarManager = RPMAvatarManager;