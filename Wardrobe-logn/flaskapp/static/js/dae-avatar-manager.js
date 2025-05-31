/**
 * DAE-Only MakeHuman Avatar Manager
 * Save as: static/js/dae-avatar-manager.js
 *
 * Simple, focused DAE loader that works with your existing avatar.html
 */

class DAEAvatarManager {
    constructor(options = {}) {
        this.container = options.container || document.getElementById('avatar-container');
        this.debug = options.debug || false;

        // Avatar state
        this.avatarModel = null;
        this.clothingItems = new Map();

        // Avatar configuration
        this.config = {
            gender: 'female',
            bodySize: 'm',
            height: 'medium',
            skinColor: 'light',
            hairType: 'short',
            hairColor: 'brown',
            eyeColor: 'brown'
        };

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Loaders
        this.colladaLoader = null;
        this.textureLoader = null;

        // Initialize
        this.init();
    }

    init() {
        console.log('🤖 Initializing DAE Avatar Manager...');
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.setupControls();
        this.setupLoaders();
        this.setupGround();
        this.animate();

        // Load default avatar
        this.loadDefaultAvatar();

        console.log('✅ DAE Avatar Manager initialized');
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
        this.scene.fog = new THREE.Fog(0xffffff, 8, 20);
    }

    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(0, 1.5, 3);
        this.camera.lookAt(0, 1, 0);
    }

    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // Optimized for DAE textures
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        // Clear existing canvas and add new one
        const existingCanvas = this.container.querySelector('canvas');
        if (existingCanvas) {
            existingCanvas.remove();
        }
        this.container.appendChild(this.renderer.domElement);
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        // Key light
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
        keyLight.position.set(5, 10, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        this.scene.add(keyLight);

        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);
    }

    setupControls() {
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.1;
            this.controls.maxPolarAngle = Math.PI * 0.8;
            this.controls.minDistance = 1;
            this.controls.maxDistance = 10;
            this.controls.target.set(0, 1, 0);
        }
    }

    setupLoaders() {
        this.textureLoader = new THREE.TextureLoader();

        if (typeof THREE.ColladaLoader !== 'undefined') {
            this.colladaLoader = new THREE.ColladaLoader();
            console.log('✅ ColladaLoader available');
        } else {
            console.error('❌ ColladaLoader not available');
            throw new Error('ColladaLoader is required for DAE files');
        }
    }

    setupGround() {
        const groundGeometry = new THREE.CircleGeometry(8, 32);
        const groundMaterial = new THREE.MeshLambertMaterial({
            color: 0xf5f5f5,
            transparent: true,
            opacity: 0.3
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    async loadDefaultAvatar() {
        console.log('🤖 Loading default DAE avatar...');
        try {
            await this.loadAvatarFromConfig(this.config);
            console.log('✅ Default avatar loaded successfully');
        } catch (error) {
            console.error('❌ Failed to load default avatar:', error);
            this.loadFallbackAvatar();
        }
    }

    async loadAvatarFromConfig(config) {
        const avatarPath = this.getAvatarPath(config);
        return this.loadDAEAvatar(avatarPath);
    }

    getAvatarPath(config) {
        const { gender, bodySize, height } = config;

        // Map to your actual DAE file structure
        // Based on your success with m_medium.dae
        let filename;

        if (gender === 'female') {
            // Your DAE files seem to be named like m_medium, l_medium, etc.
            filename = `${bodySize}_${height}`;
        } else {
            // For male, try similar pattern
            filename = `m_${bodySize}_${height}`;
        }

        return `/static/models/makehuman/bodies/${gender}/${filename}`;
    }

    async loadDAEAvatar(basePath) {
        console.log(`🔄 Loading DAE avatar from: ${basePath}`);

        try {
            // Remove existing avatar
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel = null;
            }

            // Try different possible DAE filenames
            const possiblePaths = [
                `${basePath}.dae`,
                `${basePath.replace('_medium', '')}.dae`,
                '/static/models/makehuman/bodies/female/m_medium.dae', // Your working file
                '/static/models/makehuman/bodies/female/l_medium.dae',
                '/static/models/makehuman/bodies/female/s_medium.dae'
            ];

            let loadedSuccessfully = false;

            for (const daePath of possiblePaths) {
                try {
                    console.log(`🔍 Trying to load: ${daePath}`);
                    this.avatarModel = await this.loadDAE(daePath);
                    loadedSuccessfully = true;
                    console.log(`✅ Successfully loaded: ${daePath}`);
                    break;
                } catch (error) {
                    console.warn(`⚠️ Failed to load ${daePath}:`, error.message);
                    continue;
                }
            }

            if (!loadedSuccessfully) {
                throw new Error('No valid DAE files found for this configuration');
            }

            // Configure the loaded avatar
            this.setupAvatarModel();
            this.scene.add(this.avatarModel);

            console.log('✅ DAE avatar loaded successfully');
            return this.avatarModel;

        } catch (error) {
            console.error('❌ Error loading DAE avatar:', error);
            throw error;
        }
    }

    async loadDAE(daePath) {
        return new Promise((resolve, reject) => {
            console.log('🔄 Loading DAE from:', daePath);

            this.colladaLoader.load(
                daePath,
                (collada) => {
                    console.log('✅ DAE loaded successfully');

                    // Extract the scene from Collada
                    const model = collada.scene;

                    // Process materials and textures
                    this.processDAEMaterials(model, daePath);

                    resolve(model);
                },
                (progress) => {
                    if (this.debug) {
                        console.log('📊 DAE loading progress:', progress);
                    }
                },
                (error) => {
                    console.error('❌ DAE loading error:', error);
                    reject(error);
                }
            );
        });
    }

    processDAEMaterials(model, daePath) {
        const basePath = daePath.substring(0, daePath.lastIndexOf('/') + 1);
        const texturesPath = basePath + 'textures/';

        console.log('🎨 Processing DAE materials...');
        console.log('🎨 Textures path:', texturesPath);

        model.traverse((child) => {
            if (child.isMesh) {
                console.log('🔍 Processing mesh:', child.name || 'unnamed');

                if (child.material) {
                    this.enhanceDAEMaterial(child.material, texturesPath);
                } else {
                    // Create default material
                    child.material = new THREE.MeshLambertMaterial({
                        color: this.getSkinColor(this.config.skinColor),
                        side: THREE.DoubleSide
                    });
                }

                // Ensure shadows
                child.castShadow = true;
                child.receiveShadow = true;
            }
        });
    }

    enhanceDAEMaterial(material, texturesPath) {
        console.log('🔧 Enhancing material:', material.name || 'unnamed');

        // Set up material properties
        material.side = THREE.DoubleSide;
        material.needsUpdate = true;

        // Enhanced texture handling
        if (material.map && material.map.image) {
            console.log('✅ Material has texture:', material.map.image.src);
            material.map.needsUpdate = true;
            material.map.wrapS = THREE.RepeatWrapping;
            material.map.wrapT = THREE.RepeatWrapping;
            material.map.colorSpace = THREE.SRGBColorSpace;
            material.map.flipY = false;
        } else {
            console.log('⚠️ Material missing texture, loading skin texture...');
            this.loadSkinTextureForMaterial(material, texturesPath);
        }
    }

    loadSkinTextureForMaterial(material, texturesPath) {
        // Get appropriate skin texture based on current config
        const skinTextures = {
            'light': 'young_lightskinned_female_diffuse.png',
            'medium': 'young_mediumskinned_female_diffuse.png',
            'tan': 'young_tanskinned_female_diffuse.png',
            'dark': 'young_darkskinned_female_diffuse.png'
        };

        const textureFile = skinTextures[this.config.skinColor] || skinTextures['light'];
        const texturePath = texturesPath + textureFile;

        console.log('🔍 Loading skin texture:', texturePath);

        material.map = this.textureLoader.load(
            texturePath,
            (texture) => {
                console.log('✅ Skin texture loaded:', texturePath);
                texture.needsUpdate = true;
                texture.wrapS = THREE.RepeatWrapping;
                texture.wrapT = THREE.RepeatWrapping;
                texture.colorSpace = THREE.SRGBColorSpace;
                texture.flipY = false;
                material.needsUpdate = true;
            },
            undefined,
            (error) => {
                console.log(`ℹ️ Texture not found: ${textureFile}, using color fallback`);
                material.color.setHex(this.getSkinColor(this.config.skinColor));
            }
        );
    }

    getSkinColor(skinType) {
        const skinColors = {
            'light': 0xfdbcb4,
            'medium': 0xee9b82,
            'tan': 0xd08b5b,
            'dark': 0xae5d29,
            'darker': 0x8b4513,
            'darkest': 0x654321,
            'pink': 0xffb6c1,
            'olive': 0xddbea9
        };
        return skinColors[skinType] || skinColors['light'];
    }

    setupAvatarModel() {
        if (!this.avatarModel) return;

        // Calculate bounds and scale
        const box = new THREE.Box3().setFromObject(this.avatarModel);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        // Scale to appropriate size
        let scale = 1;
        if (size.y > 10) {
            scale = 1.8 / size.y;
        } else if (size.y < 0.1) {
            scale = 1.8 / size.y;
        }

        this.avatarModel.scale.setScalar(scale);

        // Position at ground level
        this.avatarModel.position.y = -box.min.y * scale;
        this.avatarModel.position.x = -center.x * scale;
        this.avatarModel.position.z = -center.z * scale;

        console.log(`📏 Avatar scaled by ${scale.toFixed(2)}, positioned at (${this.avatarModel.position.x.toFixed(2)}, ${this.avatarModel.position.y.toFixed(2)}, ${this.avatarModel.position.z.toFixed(2)})`);
    }

    loadFallbackAvatar() {
        console.log('🔄 Loading fallback avatar...');

        const geometry = new THREE.CapsuleGeometry(0.3, 1.4, 4, 8);
        const material = new THREE.MeshLambertMaterial({
            color: this.getSkinColor(this.config.skinColor)
        });

        this.avatarModel = new THREE.Mesh(geometry, material);
        this.avatarModel.position.y = 0.9;
        this.avatarModel.castShadow = true;
        this.avatarModel.receiveShadow = true;

        this.scene.add(this.avatarModel);
        console.log('✅ Fallback avatar created');
    }

    // Avatar customization methods (compatible with avatar.html)
    async updateGender(gender) {
        this.config.gender = gender;
        await this.loadAvatarFromConfig(this.config);
    }

    async updateBodySize(bodySize) {
        this.config.bodySize = bodySize;
        await this.loadAvatarFromConfig(this.config);
    }

    async updateHeight(height) {
        this.config.height = height;
        await this.loadAvatarFromConfig(this.config);
    }

    updateSkinColor(skinColor) {
        this.config.skinColor = skinColor;

        if (this.avatarModel) {
            const color = this.getSkinColor(skinColor);

            this.avatarModel.traverse((child) => {
                if (child.isMesh && child.material) {
                    // Update color for materials without textures
                    if (!child.material.map) {
                        child.material.color.setHex(color);
                    }
                    child.material.needsUpdate = true;
                }
            });
        }
    }

    updateEyeColor(eyeColor) {
        this.config.eyeColor = eyeColor;
        console.log(`👁️ Eye color set to: ${eyeColor}`);
    }

    updateHairType(hairType) {
        this.config.hairType = hairType;
        console.log(`✂️ Hair type set to: ${hairType}`);
    }

    updateHairColor(hairColor) {
        this.config.hairColor = hairColor;
        console.log(`🎨 Hair color set to: ${hairColor}`);
    }

    // Utility methods (compatible with avatar.html)
    getConfiguration() {
        return { ...this.config };
    }

    async setConfiguration(newConfig) {
        this.config = { ...this.config, ...newConfig };
        await this.loadAvatarFromConfig(this.config);
    }

    clearAllClothing() {
        this.clothingItems.forEach((item, id) => {
            this.removeClothing(id);
        });
        this.clothingItems.clear();
    }

    removeClothing(itemId) {
        const item = this.clothingItems.get(itemId);
        if (item && item.model) {
            this.scene.remove(item.model);
            this.clothingItems.delete(itemId);
            return true;
        }
        return false;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    onWindowResize() {
        if (!this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    cleanup() {
        if (this.renderer) {
            this.renderer.dispose();
        }

        console.log('🧹 DAE Avatar Manager cleaned up');
    }
}

// Make globally available (replaces MakeHumanAvatarManager)
window.MakeHumanAvatarManager = DAEAvatarManager;
window.DAEAvatarManager = DAEAvatarManager;