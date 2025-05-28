// MakeHuman Avatar Management System
// File: static/js/makehuman-avatar.js

class MakeHumanAvatarManager {
    constructor(options = {}) {
        this.container = options.container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Avatar components
        this.currentBody = null;
        this.currentHair = null;
        this.hairMeshes = new Map(); // Cache hair models
        this.bodyMeshes = new Map(); // Cache body models

        // Current configuration
        this.config = {
            gender: 'female',
            bodySize: 'm',
            height: 'medium',
            hairType: 'short',
            skinColor: 'light',
            hairColor: 'brown',
            eyeColor: 'brown'
        };

        this.loader = new THREE.GLTFLoader();
        this.textureLoader = new THREE.TextureLoader();

        this.init();
    }

    init() {
        this.setupScene();
        this.setupLighting();
        this.setupCamera();
        this.setupControls();
        this.animate();

        // Load default avatar
        this.loadAvatar();
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf5f7fa);

        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(
            this.container.clientWidth,
            this.container.clientHeight
        );
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;

        this.container.appendChild(this.renderer.domElement);

        // Handle resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Main directional light
        const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
        mainLight.position.set(5, 10, 5);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 2048;
        mainLight.shadow.mapSize.height = 2048;
        this.scene.add(mainLight);

        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);
    }

    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(
            45,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 3);
    }

    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.target.set(0, 0, 0);
        this.controls.maxPolarAngle = Math.PI * 0.9;
        this.controls.minDistance = 1.5;
        this.controls.maxDistance = 8;
    }

    async loadAvatar() {
        try {
            await this.loadBody();
            await this.loadHair();
            this.applyTextures();
            console.log('✅ MakeHuman avatar loaded successfully');
        } catch (error) {
            console.error('❌ Error loading avatar:', error);
        }
    }

    async loadBody() {
        const bodyKey = `${this.config.gender}_${this.config.bodySize}_${this.config.height}`;

        // Check cache first
        if (this.bodyMeshes.has(bodyKey)) {
            if (this.currentBody) {
                this.scene.remove(this.currentBody);
            }
            this.currentBody = this.bodyMeshes.get(bodyKey).clone();
            this.scene.add(this.currentBody);
            return;
        }

        // Load new body model
        const bodyPath = `/static/models/makehuman/bodies/${this.config.gender}/${this.config.bodySize}_${this.config.height}.glb`;

        try {
            const gltf = await this.loadModel(bodyPath);

            // Remove old body
            if (this.currentBody) {
                this.scene.remove(this.currentBody);
            }

            this.currentBody = gltf.scene;
            this.currentBody.scale.setScalar(1);
            this.currentBody.position.set(0, -1, 0);

            // Cache the model
            this.bodyMeshes.set(bodyKey, this.currentBody.clone());

            this.scene.add(this.currentBody);

        } catch (error) {
            console.error(`❌ Failed to load body: ${bodyPath}`, error);
            throw error;
        }
    }

    async loadHair() {
        if (this.config.hairType === 'bald') {
            if (this.currentHair) {
                this.scene.remove(this.currentHair);
                this.currentHair = null;
            }
            return;
        }

        const hairKey = this.config.hairType;

        // Check cache first
        if (this.hairMeshes.has(hairKey)) {
            if (this.currentHair) {
                this.scene.remove(this.currentHair);
            }
            this.currentHair = this.hairMeshes.get(hairKey).clone();
            this.scene.add(this.currentHair);
            this.applyHairColor();
            return;
        }

        // Load new hair model
        const hairPath = `/static/models/makehuman/hair/${this.config.hairType}.glb`;

        try {
            const gltf = await this.loadModel(hairPath);

            // Remove old hair
            if (this.currentHair) {
                this.scene.remove(this.currentHair);
            }

            this.currentHair = gltf.scene;
            this.currentHair.scale.setScalar(1);
            this.currentHair.position.set(0, -1, 0);

            // Cache the model
            this.hairMeshes.set(hairKey, this.currentHair.clone());

            this.scene.add(this.currentHair);
            this.applyHairColor();

        } catch (error) {
            console.error(`❌ Failed to load hair: ${hairPath}`, error);
            // Don't throw - hair is optional
        }
    }

    applyTextures() {
        this.applySkinColor();
        this.applyEyeColor();
    }

    applySkinColor() {
        if (!this.currentBody) return;

        const skinTexturePath = `/static/models/makehuman/textures/skin/${this.config.skinColor}.jpg`;

        this.textureLoader.load(skinTexturePath, (texture) => {
            this.currentBody.traverse((child) => {
                if (child.isMesh && child.material) {
                    // Apply to skin materials (usually diffuse/albedo)
                    if (child.material.map || child.material.name.includes('skin')) {
                        const material = child.material.clone();
                        material.map = texture;
                        material.needsUpdate = true;
                        child.material = material;
                    }
                }
            });
        }, undefined, (error) => {
            console.warn(`⚠️ Could not load skin texture: ${skinTexturePath}`, error);
            // Fallback to color-only
            this.applySkinColorFallback();
        });
    }

    applySkinColorFallback() {
        if (!this.currentBody) return;

        const colorMap = {
            'light': 0xFDBCB4,
            'medium': 0xEE9B82,
            'tan': 0xD08B5B,
            'dark': 0xAE5D29,
            'darker': 0x8B4513,
            'darkest': 0x654321,
            'pink': 0xFFB6C1,
            'olive': 0xDDBEA9
        };

        const skinColor = new THREE.Color(colorMap[this.config.skinColor] || 0xFDBCB4);

        this.currentBody.traverse((child) => {
            if (child.isMesh && child.material) {
                const material = child.material.clone();
                material.color = skinColor;
                material.needsUpdate = true;
                child.material = material;
            }
        });
    }

    applyHairColor() {
        if (!this.currentHair) return;

        const hairTexturePath = `/static/models/makehuman/textures/hair/${this.config.hairColor}.jpg`;

        this.textureLoader.load(hairTexturePath, (texture) => {
            this.currentHair.traverse((child) => {
                if (child.isMesh && child.material) {
                    const material = child.material.clone();
                    material.map = texture;
                    material.needsUpdate = true;
                    child.material = material;
                }
            });
        }, undefined, (error) => {
            console.warn(`⚠️ Could not load hair texture: ${hairTexturePath}`, error);
            // Fallback to color-only approach
            this.applyHairColorFallback();
        });
    }

    applyHairColorFallback() {
        if (!this.currentHair) return;

        const colorMap = {
            'brown': 0x8B4513,
            'black': 0x000000,
            'blonde': 0xFFD700,
            'red': 0xDC143C,
            'auburn': 0xA0522D,
            'gray': 0xC0C0C0,
            'white': 0xFFFFFF,
            'pink': 0xFF69B4
        };

        const color = new THREE.Color(colorMap[this.config.hairColor] || 0x8B4513);

        this.currentHair.traverse((child) => {
            if (child.isMesh && child.material) {
                const material = child.material.clone();
                material.color = color;
                material.needsUpdate = true;
                child.material = material;
            }
        });
    }

    applyEyeColor() {
        if (!this.currentBody) return;

        const colorMap = {
            'brown': 0x6B4423,
            'blue': 0x4A90E2,
            'green': 0x50C878,
            'light-blue': 0x87CEEB,
            'purple': 0x800080,
            'gray': 0x708090,
            'dark-green': 0x228B22,
            'hazel': 0xB8860B
        };

        const eyeColor = new THREE.Color(colorMap[this.config.eyeColor] || 0x6B4423);

        this.currentBody.traverse((child) => {
            if (child.isMesh && child.material &&
                (child.name.includes('eye') || child.material.name.includes('eye'))) {
                const material = child.material.clone();
                material.color = eyeColor;
                material.needsUpdate = true;
                child.material = material;
            }
        });
    }

    loadModel(path) {
        return new Promise((resolve, reject) => {
            this.loader.load(path, resolve, undefined, reject);
        });
    }

    // Configuration update methods
    async updateGender(gender) {
        this.config.gender = gender;
        await this.loadBody();
        this.applyTextures();
    }

    async updateBodySize(size) {
        this.config.bodySize = size;
        await this.loadBody();
        this.applyTextures();
    }

    async updateHeight(height) {
        this.config.height = height;
        await this.loadBody();
        this.applyTextures();
    }

    async updateHairType(hairType) {
        this.config.hairType = hairType;
        await this.loadHair();
    }

    updateSkinColor(skinColor) {
        this.config.skinColor = skinColor;
        this.applySkinColor();
    }

    updateHairColor(hairColor) {
        this.config.hairColor = hairColor;
        this.applyHairColor();
    }

    updateEyeColor(eyeColor) {
        this.config.eyeColor = eyeColor;
        this.applyEyeColor();
    }

    // Utility methods
    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    getConfiguration() {
        return { ...this.config };
    }

    async setConfiguration(newConfig) {
        const oldConfig = { ...this.config };
        this.config = { ...this.config, ...newConfig };

        // Reload components that changed
        if (oldConfig.gender !== this.config.gender ||
            oldConfig.bodySize !== this.config.bodySize ||
            oldConfig.height !== this.config.height) {
            await this.loadBody();
        }

        if (oldConfig.hairType !== this.config.hairType) {
            await this.loadHair();
        }

        this.applyTextures();
    }
}

// Make it globally available
window.MakeHumanAvatarManager = MakeHumanAvatarManager;