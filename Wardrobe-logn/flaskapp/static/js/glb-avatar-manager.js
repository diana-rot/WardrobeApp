/**
 * CLEANED GLB Avatar Manager - Removed Unused Paths & Fixed Configuration
 * Save as: static/js/glb-avatar-manager.js
 */

class CustomizableGLBAvatarManager {
    constructor(options = {}) {
        this.container = options.container || document.getElementById('avatar-container');
        this.debug = options.debug || false;

        // Avatar state
        this.avatarModel = null;
        this.clothingItems = new Map();
        this.currentHairModel = null;

        // Loading state protection
        this.isLoadingAvatar = false;
        this.isLoadingHair = false;
        this.loadingQueue = new Set();
        this.isInitialized = false;
        this.hasLoadedDefault = false;

        // Avatar configuration
        this.config = {
            gender: 'female',
            bodySize: 'm',
            height: 'medium',
            skinColor: 'light',
            hairType: 'elvis_hazel', // Changed to first available hair style
            hairColor: 'brown',
            eyeColor: 'brown'
        };

        // FIXED: Your EXACT coordinates for each avatar height
        this.hairPositionsByHeight = {
            'short': { x: 0, y: -0.534, z: -0.13, scale: 1.0 },
            'medium': { x: 0, y: -0.39, z: -0.12, scale: 1.0 },
            'long': { x: 0, y: -0.534, z: -0.13, scale: 1.0 }
        };

        // CLEANED: Hair styles configuration - removed unused paths
        this.hairStyles = {
            female: {
                'elvis_hazel': {
                    name: 'Elvis Hazel',
                    preview: '/static/models/makehuman/hair/previews/elvis_hazel.jpg',
                    glbPath: '/static/models/makehuman/hair/Elvs_Hazel_Hair/elvs_hazel.glb',
                    category: 'medium'
                },
                'french_bob': {
                    name: 'French Bob',
                    preview: '/static/models/makehuman/hair/previews/bob.jpg',
                    glbPath: '/static/models/makehuman/hair/French_Bob_Blonde/bob.glb',
                    category: 'short'
                },
                'ponytail': {
                    name: 'PonyTail',
                    preview: '/static/models/makehuman/hair/previews/ponytail.jpg',
                    glbPath: '/static/models/makehuman/hair/Hair_06/ponytail.glb',
                    category: 'medium'
                },
                'hair_07': {
                    name: 'Wavy Medium',
                    preview: '/static/models/makehuman/hair/previews/hair_07.jpg',
                    glbPath: '/static/models/makehuman/hair/Hair_07/hair_07.glb',
                    category: 'medium'
                },
                'hair_08': {
                    name: 'Long Straight',
                    preview: '/static/models/makehuman/hair/previews/hair_08.jpg',
                    glbPath: '/static/models/makehuman/hair/Hair_08/hair_08.glb',
                    category: 'long'
                },
                'hair1': {
                    name: 'Short Curly',
                    preview: '/static/models/makehuman/hair/previews/hair1.jpg',
                    glbPath: '/static/models/makehuman/hair/hair1/hair1.glb',
                    category: 'short'
                },
                'short_strawberry': {
                    name: 'Strawberry Hair',
                    preview: '/static/models/makehuman/hair/previews/strawberry_hair.jpg',
                    glbPath:  '/static/models/makehuman/hair/Strawberry_Cloud_Hair/strawberry.glb',
                    category: 'medium'
                },
                  'bun': {
                    name: 'Bun',
                    preview: '/static/models/makehuman/hair/previews/bun_blonde.jpg',
                    glbPath:  '/static/models/makehuman/hair/Wig_bun_blonde_female_braids/bun.glb',
                    category: 'medium'
                }
            },
            male: {
                'male_short': {
                    name: 'Short Male',
                    preview: '/static/models/makehuman/hair/previews/male_short.jpg',
                    glbPath: '/static/models/makehuman/hair/male_short_hair/male_short_hair.glb',
                    category: 'short'
                },
                'bald': {
                    name: 'Bald',
                    preview: '/static/models/makehuman/hair/previews/bald.jpg',
                    glbPath: null, // No model for bald
                    category: 'none'
                }
            }
        };

        // EXPANDED: Eye texture mapping with more options
        this.eyeTextures = {
            'brown': '/static/models/makehuman/bodies/female/textures/brown_eye.png',
            'blue': '/static/models/makehuman/bodies/female/textures/blue_eye.png',
            'green': '/static/models/makehuman/bodies/female/textures/green_eye.png',
            'light-blue': '/static/models/makehuman/bodies/female/textures/light_blue_eye.png',
            'purple': '/static/models/makehuman/bodies/female/textures/purple_eye.png',
            'gray': '/static/models/makehuman/bodies/female/textures/gray_eye.png',
            'grey': '/static/models/makehuman/bodies/female/textures/gray_eye.png',
            'dark-green': '/static/models/makehuman/bodies/female/textures/dark_green_eye.png',
            'hazel': '/static/models/makehuman/bodies/female/textures/hazel_eye.png'
        };

        // Loaded textures cache
        this.loadedTextures = new Map();

        // EXPANDED: More diverse skin color options
        this.skinColors = {
            'light': 0xfdbcb4,
            'fair': 0xfce4ec,
            'pale': 0xf8bbd9,
            'medium': 0xee9b82,
            'olive': 0xddbea9,
            'tan': 0xd08b5b,
            'bronze': 0xcd7f32,
            'dark': 0xae5d29,
            'darker': 0x8b4513,
            'darkest': 0x654321,
            'ebony': 0x3c2414,
            'pink': 0xffb6c1,
            'warm': 0xdeb887,
            'cool': 0xd3d3d3
        };

        // EXPANDED: More eye color variations
        this.eyeColors = {
            'brown': 0x6B4423,
            'dark-brown': 0x4A2C17,
            'light-brown': 0x8B6914,
            'blue': 0x4A90E2,
            'light-blue': 0x87CEEB,
            'dark-blue': 0x0F4C75,
            'sky-blue': 0x87CEFA,
            'green': 0x50C878,
            'light-green': 0x90EE90,
            'dark-green': 0x228B22,
            'emerald': 0x50C878,
            'hazel': 0xB8860B,
            'amber': 0xFFBF00,
            'purple': 0x800080,
            'violet': 0x8A2BE2,
            'gray': 0x708090,
            'grey': 0x708090,
            'silver': 0xC0C0C0,
            'gold': 0xFFD700
        };

        // EXPANDED: Extensive hair color palette
        this.hairColors = {
            // Natural Browns
            'brown': 0x8B4513,
            'light-brown': 0xA0522D,
            'dark-brown': 0x654321,
            'chestnut': 0x954535,
            'chocolate': 0x7B3F00,
            'mahogany': 0xC04000,
            'coffee': 0x6F4E37,

            // Blacks
            'black': 0x000000,
            'jet-black': 0x0C0C0C,
            'charcoal': 0x36454F,

            // Blondes
            'blonde': 0xFFD700,
            'light-blonde': 0xFFF8DC,
            'dark-blonde': 0xDAA520,
            'platinum': 0xE5E4E2,
            'honey': 0xFFB347,
            'strawberry-blonde': 0xFF8C69,
            'golden': 0xFFD700,

            // Reds
            'red': 0xDC143C,
            'auburn': 0xA0522D,
            'copper': 0xB87333,
            'ginger': 0xB06500,
            'burgundy': 0x800020,
            'cherry': 0xDE3163,

            // Grays & Whites
            'gray': 0xC0C0C0,
            'grey': 0xC0C0C0,
            'silver': 0xC0C0C0,
            'white': 0xFFFFFF,
            'salt-pepper': 0x999999,

            // Fantasy Colors
            'pink': 0xFF69B4,
            'hot-pink': 0xFF1493,
            'purple': 0x800080,
            'violet': 0x8A2BE2,
            'blue': 0x0000FF,
            'teal': 0x008080,
            'green': 0x008000,
            'mint': 0x98FB98,
            'orange': 0xFF8C00,
            'rainbow': 0xFF69B4 // Will cycle through colors
        };

        // Material references
        this.avatarMaterials = {
            skin: [],
            eyes: [],
            hair: [],
            underwear: []
        };

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.gltfLoader = null;
        this.textureLoader = null;

        // Initialize
        this.init();
    }

    init() {
        console.log('🤖 Initializing CLEANED GLB Avatar Manager...');
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.setupControls();
        this.setupLoaders();
        this.setupGround();
        this.animate();

        this.isInitialized = true;
        console.log('✅ CLEANED GLB Avatar Manager initialized');
        console.log('🎯 Using EXACT coordinates for perfect hair positioning');
    }

    // Manual method to load default avatar
    async loadDefaultAvatarManually() {
        if (this.hasLoadedDefault || this.isLoadingAvatar) {
            console.log('⚠️ Default avatar already loaded or loading');
            return;
        }

        console.log('🤖 Loading default GLB avatar with PERFECT hair positioning...');
        this.hasLoadedDefault = true;

        try {
            await this.loadAvatarFromConfig(this.config);
            console.log('✅ Default avatar loaded successfully');

            // Load default hair with PERFECT positioning
            setTimeout(async () => {
                if (!this.isLoadingHair && !this.currentHairModel) {
                    await this.updateHairStyle(this.config.hairType);
                }
            }, 1000);
        } catch (error) {
            console.error('❌ Failed to load default avatar:', error);
            this.hasLoadedDefault = false;
            this.loadFallbackAvatar();
        }
    }

    // Loading protection methods
    async loadAvatarFromConfig(config) {
        if (this.isLoadingAvatar) {
            console.log('⚠️ Avatar loading already in progress');
            return null;
        }

        const avatarPath = this.getAvatarPath(config);

        if (this.loadingQueue.has(avatarPath)) {
            console.log('⚠️ Avatar path already queued:', avatarPath);
            return null;
        }

        return this.loadGLBAvatar(avatarPath);
    }

    getAvatarPath(config) {
        const { gender, bodySize, height } = config;
        const fileName = `${bodySize}_${height}`;
        const fullPath = `/static/models/makehuman/bodies/${gender}/${fileName}.glb`;
        return fullPath;
    }

    // Avatar loading with protection
    async loadGLBAvatar(glbPath) {
        if (this.isLoadingAvatar) {
            console.log('⚠️ Avatar loading blocked - already in progress');
            return null;
        }

        if (this.loadingQueue.has(glbPath)) {
            console.log('⚠️ Avatar loading blocked - already queued:', glbPath);
            return null;
        }

        console.log(`🔄 Loading GLB avatar from: ${glbPath}`);

        this.isLoadingAvatar = true;
        this.loadingQueue.add(glbPath);

        try {
            // Clear existing avatar
            if (this.avatarModel) {
                this.scene.remove(this.avatarModel);
                this.avatarModel = null;
            }

            // Reset materials
            this.avatarMaterials = {
                skin: [],
                eyes: [],
                hair: [],
                underwear: []
            };

            // Load new avatar
            this.avatarModel = await this.loadGLB(glbPath);
            this.setupAvatarModel();
            this.scene.add(this.avatarModel);

            // Apply colors
            this.updateSkinColor(this.config.skinColor);
            await this.updateEyeColor(this.config.eyeColor);

            console.log('✅ GLB avatar loaded successfully');
            return this.avatarModel;

        } catch (error) {
            console.error('❌ Error loading GLB avatar:', error);
            throw error;
        } finally {
            this.isLoadingAvatar = false;
            this.loadingQueue.delete(glbPath);
            console.log('🔓 Avatar loading flags cleared');
        }
    }

    async loadGLB(glbPath) {
        return new Promise((resolve, reject) => {
            this.gltfLoader.load(
                glbPath,
                (gltf) => {
                    const model = gltf.scene;
                    this.processGLBModel(model);
                    resolve(model);
                },
                (progress) => {
                    if (this.debug) {
                        const percentage = (progress.loaded / progress.total * 100).toFixed(1);
                        console.log(`📊 GLB loading progress: ${percentage}%`);
                    }
                },
                (error) => {
                    console.error('❌ GLB loading error:', error);
                    reject(error);
                }
            );
        });
    }

    // FIXED: Hair loading with PERFECT positioning using your exact coordinates
    async updateHairStyle(hairStyleKey) {
        if (this.isLoadingHair) {
            console.log('⚠️ Hair loading blocked - already in progress');
            return false;
        }

        console.log(`🦱 Updating hair style to: ${hairStyleKey} (using PERFECT coordinates)`);

        const gender = this.config.gender || 'female';
        const hairData = this.hairStyles[gender][hairStyleKey];

        if (!hairData) {
            console.error(`❌ Hair style not found: ${hairStyleKey} for gender: ${gender}`);
            return false;
        }

        this.isLoadingHair = true;

        try {
            // Remove existing hair
            this.removeCurrentHair();

            // Handle bald case
            if (!hairData.glbPath || hairStyleKey === 'bald') {
                console.log('👩‍🦲 Applied bald style (no hair model)');
                this.config.hairType = hairStyleKey;
                return true;
            }

            console.log(`🦱 Loading GLB hair model: ${hairData.name}`);
            const hairModel = await this.loadHairGLB(hairData.glbPath);

            if (hairModel) {
                // Apply PERFECT positioning with your exact coordinates
                this.applyPerfectHairPositioning(hairModel);

                // Apply materials and add to scene
                this.enhanceHairMaterials(hairModel);
                this.scene.add(hairModel);
                this.currentHairModel = hairModel;
                this.currentHairModel.name = `hair_${hairStyleKey}`;
                this.config.hairType = hairStyleKey;

                console.log(`✅ GLB hair model loaded with PERFECT positioning`);
                const coords = this.hairPositionsByHeight[this.config.height];
                console.log(`📍 Hair positioned at: (${coords.x}, ${coords.y}, ${coords.z}) scale: ${coords.scale}x`);
                return true;
            }
        } catch (error) {
            console.error(`❌ Failed to load GLB hair model: ${error.message}`);
            return false;
        } finally {
            this.isLoadingHair = false;
            console.log('🔓 Hair loading flag cleared');
        }

        return false;
    }

    // FIXED: Perfect hair positioning using your EXACT coordinates
    applyPerfectHairPositioning(hairModel) {
        if (!this.avatarModel || !hairModel) {
            console.warn('❌ Missing avatar or hair model for positioning');
            return false;
        }

        console.log('🎯 Applying PERFECT hair positioning with your EXACT coordinates...');

        try {
            // Get the exact coordinates for current avatar height
            const heightKey = this.config.height || 'medium';
            const coords = this.hairPositionsByHeight[heightKey];

            if (!coords) {
                console.error(`❌ No coordinates found for height: ${heightKey}`);
                return false;
            }

            // Apply your exact coordinates
            hairModel.scale.setScalar(coords.scale);
            hairModel.position.set(coords.x, coords.y, coords.z);
            hairModel.rotation.set(0, 0, 0);

            console.log(`✅ PERFECT hair positioning complete for ${heightKey} avatar:`);
            console.log(`   Position: (${coords.x}, ${coords.y}, ${coords.z})`);
            console.log(`   Scale: ${coords.scale}x`);

            return true;

        } catch (error) {
            console.error('❌ Perfect hair positioning failed:', error);
            return false;
        }
    }

    // Manual hair position adjustment (simplified)
    adjustHairPositionManually(adjustments = {}) {
        if (!this.currentHairModel) {
            console.log('❌ No hair model available for manual adjustment');
            return false;
        }

        const {
            scaleMultiplier = 1.0,
            moveDown = 0,
            moveBack = 0,
            moveLeft = 0
        } = adjustments;

        const hairModel = this.currentHairModel;

        // Apply adjustments
        if (scaleMultiplier !== 1.0) {
            const newScale = hairModel.scale.x * scaleMultiplier;
            hairModel.scale.setScalar(newScale);
        }

        if (moveDown !== 0) hairModel.position.y -= moveDown;
        if (moveBack !== 0) hairModel.position.z -= moveBack;
        if (moveLeft !== 0) hairModel.position.x -= moveLeft;

        console.log(`🔧 Hair manually adjusted`);
        return true;
    }

    // Scene setup methods
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
    }

    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
        this.camera.position.set(0, 1.6, 4);
        this.camera.lookAt(0, 1.2, 0);
    }

    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false
        });

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        const existingCanvas = this.container.querySelector('canvas');
        if (existingCanvas) {
            existingCanvas.remove();
        }
        this.container.appendChild(this.renderer.domElement);
    }

    setupLighting() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
        keyLight.position.set(-5, 10, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -10;
        keyLight.shadow.camera.right = 10;
        keyLight.shadow.camera.top = 10;
        keyLight.shadow.camera.bottom = -10;
        keyLight.shadow.bias = -0.0001;
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
        fillLight.position.set(5, 5, 5);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xffffff, 0.8);
        rimLight.position.set(0, 5, -8);
        this.scene.add(rimLight);

        const topLight = new THREE.DirectionalLight(0xffffff, 0.4);
        topLight.position.set(0, 10, 0);
        this.scene.add(topLight);

        const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
        hemisphereLight.position.set(0, 20, 0);
        this.scene.add(hemisphereLight);

        const eyeLight = new THREE.DirectionalLight(0xffffff, 0.3);
        eyeLight.position.set(0, 2, 3);
        eyeLight.name = 'eyeLight';
        this.scene.add(eyeLight);
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

        if (typeof THREE.GLTFLoader !== 'undefined') {
            this.gltfLoader = new THREE.GLTFLoader();
            console.log('✅ GLTFLoader available');
        } else {
            console.error('❌ GLTFLoader not available');
            throw new Error('GLTFLoader is required for GLB files');
        }
    }

    setupGround() {
        const gridHelper = new THREE.GridHelper(20, 20, 0xcccccc, 0xdddddd);
        gridHelper.material.opacity = 0.8;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);

        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.ShadowMaterial({
            opacity: 0.15,
            color: 0x000000
        });

        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    // Hair loading methods
    async loadHairGLB(glbPath) {
        console.log(`🔄 Loading GLB hair from: ${glbPath}`);

        return new Promise((resolve, reject) => {
            this.gltfLoader.load(
                glbPath,
                (gltf) => {
                    console.log('✅ GLB hair loaded successfully');
                    const hairModel = gltf.scene;
                    this.processHairModel(hairModel);
                    resolve(hairModel);
                },
                (progress) => {
                    if (this.debug) {
                        const percentage = (progress.loaded / progress.total * 100).toFixed(1);
                        console.log(`📊 GLB hair loading progress: ${percentage}%`);
                    }
                },
                (error) => {
                    console.error('❌ GLB hair loading error:', error);
                    reject(error);
                }
            );
        });
    }

    processHairModel(hairModel) {
        console.log('🎨 Processing GLB hair model...');

        hairModel.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = false;
                child.frustumCulled = false;

                if (child.material) {
                    if (child.material.map) {
                        child.material.map.colorSpace = THREE.SRGBColorSpace;
                        child.material.map.flipY = false;
                    }

                    if (child.material.transparent === undefined) {
                        child.material.transparent = true;
                    }
                    if (child.material.alphaTest === undefined) {
                        child.material.alphaTest = 0.1;
                    }
                    if (child.material.side === undefined) {
                        child.material.side = THREE.DoubleSide;
                    }

                    child.material.needsUpdate = true;
                }
            }
        });
    }

    enhanceHairMaterials(hairModel) {
        console.log('🎨 Enhancing GLB hair materials...');

        const hairColor = this.hairColors[this.config.hairColor] || this.hairColors['brown'];

        hairModel.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.color = new THREE.Color(hairColor);
                child.material.transparent = true;
                child.material.alphaTest = 0.1;
                child.material.side = THREE.DoubleSide;
                child.material.depthWrite = true;
                child.material.depthTest = true;

                if (child.material.opacity > 0.5) {
                    child.material.opacity = Math.min(child.material.opacity, 0.98);
                }

                child.material.needsUpdate = true;
            }
        });

        hairModel.rotation.set(0, 0, 0);
    }

    removeCurrentHair() {
        if (this.currentHairModel) {
            this.scene.remove(this.currentHairModel);

            this.currentHairModel.traverse((child) => {
                if (child.isMesh) {
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) {
                        if (child.material.map) child.material.map.dispose();
                        child.material.dispose();
                    }
                }
            });

            this.currentHairModel = null;
            console.log('🗑️ Previous hair model removed');
        }
    }

    // Model processing methods
    processGLBModel(model) {
        console.log('🎨 Processing GLB model for customization...');

        model.traverse((child) => {
            if (child.isMesh) {
                this.categorizeMaterial(child);
                child.castShadow = true;
                child.receiveShadow = true;

                if (child.material) {
                    if (child.material.map) {
                        child.material.map.colorSpace = THREE.SRGBColorSpace;
                        child.material.map.flipY = false;
                    }
                    child.material.needsUpdate = true;
                }
            }
        });
    }

    categorizeMaterial(mesh) {
        const meshName = (mesh.name || '').toLowerCase();
        const materialName = (mesh.material.name || '').toLowerCase();
        const material = mesh.material;

        if (!material) return;

        if (meshName.includes('eye') || materialName.includes('eye') ||
            meshName.includes('head') || meshName.includes('face') ||
            materialName.includes('head') || materialName.includes('face')) {
            this.avatarMaterials.eyes.push(material);
        }
        else if (meshName.includes('body') || meshName.includes('arm') ||
                 meshName.includes('leg') || meshName.includes('torso') ||
                 meshName.includes('neck') || meshName.includes('skin') ||
                 meshName.includes('hand') || meshName.includes('foot')) {
            this.avatarMaterials.skin.push(material);
        }
        else {
            this.avatarMaterials.skin.push(material);
        }
    }

    setupAvatarModel() {
        if (!this.avatarModel) return;

        const box = new THREE.Box3().setFromObject(this.avatarModel);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        let scale = 1.0;
        if (size.y > 3) {
            scale = 1.8 / size.y;
        } else if (size.y < 1) {
            scale = 1.8 / size.y;
        }

        this.avatarModel.scale.setScalar(scale);
        this.avatarModel.position.y = -box.min.y * scale;
        this.avatarModel.position.x = -center.x * scale;
        this.avatarModel.position.z = -center.z * scale;

        console.log(`📏 Avatar scaled and positioned`);
    }

    // Color update methods
    updateSkinColor(skinColor) {
        this.config.skinColor = skinColor;
        const newColor = this.skinColors[skinColor] || this.skinColors['light'];

        this.avatarMaterials.skin.forEach((material) => {
            if (material.color) {
                material.color.setHex(newColor);
                material.needsUpdate = true;
            }
        });
    }

    async updateEyeColor(eyeColor) {
        this.config.eyeColor = eyeColor;
        const eyeTexture = await this.loadEyeTexture(eyeColor);

        if (eyeTexture) {
            this.applyEyeTexture(eyeTexture, eyeColor);
        } else {
            this.applyEyeColor(eyeColor);
        }
    }

    async loadEyeTexture(eyeColor) {
        const texturePath = this.eyeTextures[eyeColor];
        if (!texturePath) return null;

        if (this.loadedTextures.has(eyeColor)) {
            return this.loadedTextures.get(eyeColor);
        }

        return new Promise((resolve) => {
            this.textureLoader.load(
                texturePath,
                (texture) => {
                    texture.colorSpace = THREE.SRGBColorSpace;
                    texture.flipY = false;
                    this.loadedTextures.set(eyeColor, texture);
                    resolve(texture);
                },
                undefined,
                () => resolve(null)
            );
        });
    }

    applyEyeTexture(texture, eyeColor) {
        this.avatarMaterials.eyes.forEach((material) => {
            material.map = texture;
            material.needsUpdate = true;
        });
    }

    applyEyeColor(eyeColor) {
        const newColor = this.eyeColors[eyeColor] || this.eyeColors['brown'];
        this.avatarMaterials.eyes.forEach((material) => {
            if (material.color) {
                material.color.setHex(newColor);
                material.needsUpdate = true;
            }
        });
    }

    updateHairColor(hairColor) {
        this.config.hairColor = hairColor;

        if (!this.currentHairModel) return;

        const newColor = this.hairColors[hairColor] || this.hairColors['brown'];

        this.currentHairModel.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.color.setHex(newColor);
                child.material.needsUpdate = true;
            }
        });
    }

    // Avatar update methods
    async updateGender(gender) {
        console.log(`👤 Updating gender to: ${gender}`);
        this.config.gender = gender;
        this.removeCurrentHair();
        await this.loadAvatarFromConfig(this.config);

        const defaultHairStyles = {
            'female': 'elvis_hazel',
            'male': 'male_short'
        };

        const defaultHair = defaultHairStyles[gender] || 'elvis_hazel';
        setTimeout(async () => {
            await this.updateHairStyle(defaultHair);
        }, 1000);
    }

    async updateBodySize(bodySize) {
        console.log(`📏 Updating body size to: ${bodySize}`);
        this.config.bodySize = bodySize;
        await this.loadAvatarFromConfig(this.config);

        if (this.config.hairType) {
            setTimeout(async () => {
                await this.updateHairStyle(this.config.hairType);
            }, 1000);
        }
    }

    async updateHeight(height) {
        console.log(`📐 Updating height to: ${height}`);
        this.config.height = height;
        await this.loadAvatarFromConfig(this.config);

        if (this.config.hairType) {
            setTimeout(async () => {
                await this.updateHairStyle(this.config.hairType);
            }, 1000);
        }
    }

    // Hair style methods
    getAvailableHairStyles(gender = null) {
        const targetGender = gender || this.config.gender || 'female';
        return this.hairStyles[targetGender] || {};
    }

    generateHairPreviewHTML(gender = null) {
        const targetGender = gender || this.config.gender || 'female';
        const hairStyles = this.getAvailableHairStyles(targetGender);

        let html = '<div class="hair-preview-grid">';

        Object.entries(hairStyles).forEach(([styleKey, styleData]) => {
            html += `
                <div class="hair-preview-option" data-hair-style="${styleKey}">
                    <div class="hair-preview-image">
                        <img src="${styleData.preview}" 
                             alt="${styleData.name}"
                             onerror="this.parentElement.innerHTML='<div class=\\'preview-placeholder\\'>${styleData.name}</div>'">
                    </div>
                    <div class="hair-preview-name">${styleData.name}</div>
                </div>
            `;
        });

        html += '</div>';
        return html;
    }

    // Configuration methods
    getConfiguration() {
        return { ...this.config };
    }

    async setConfiguration(newConfig) {
        const oldConfig = { ...this.config };
        this.config = { ...this.config, ...newConfig };

        const majorChanges = ['gender', 'bodySize', 'height'];
        const needsReload = majorChanges.some(key =>
            oldConfig[key] !== newConfig[key] && newConfig[key] !== undefined
        );

        if (needsReload) {
            await this.loadAvatarFromConfig(this.config);

            if (newConfig.hairType || this.config.hairType) {
                setTimeout(async () => {
                    await this.updateHairStyle(newConfig.hairType || this.config.hairType);
                }, 1000);
            }
        } else {
            if (oldConfig.skinColor !== newConfig.skinColor && newConfig.skinColor) {
                this.updateSkinColor(newConfig.skinColor);
            }
            if (oldConfig.eyeColor !== newConfig.eyeColor && newConfig.eyeColor) {
                await this.updateEyeColor(newConfig.eyeColor);
            }
            if (oldConfig.hairColor !== newConfig.hairColor && newConfig.hairColor) {
                this.updateHairColor(newConfig.hairColor);
            }
            if (oldConfig.hairType !== newConfig.hairType && newConfig.hairType) {
                await this.updateHairStyle(newConfig.hairType);
            }
        }
    }

    // Clothing methods
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

    // Fallback avatar
    loadFallbackAvatar() {
        console.log('🔄 Loading fallback avatar...');

        const geometry = new THREE.CapsuleGeometry(0.3, 1.4, 4, 8);
        const material = new THREE.MeshLambertMaterial({
            color: this.skinColors[this.config.skinColor] || this.skinColors['light']
        });

        this.avatarModel = new THREE.Mesh(geometry, material);
        this.avatarModel.position.y = 0.9;
        this.avatarModel.castShadow = true;
        this.avatarModel.receiveShadow = true;

        this.scene.add(this.avatarModel);
        console.log('✅ Fallback avatar created');
    }

    // Animation loop
    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update();
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    // Window resize handler
    onWindowResize() {
        if (!this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    // Cleanup method
    cleanup() {
        this.loadedTextures.forEach((texture) => {
            texture.dispose();
        });
        this.loadedTextures.clear();

        this.removeCurrentHair();

        if (this.renderer) {
            this.renderer.dispose();
        }

        // Clear all loading states
        this.isLoadingAvatar = false;
        this.isLoadingHair = false;
        this.loadingQueue.clear();
        this.hasLoadedDefault = false;

        console.log('🧹 Avatar Manager cleaned up');
    }

    // Status check methods
    isAvatarLoading() {
        return this.isLoadingAvatar;
    }

    isHairLoading() {
        return this.isLoadingHair;
    }

    getLoadingStatus() {
        return {
            avatar: this.isLoadingAvatar,
            hair: this.isLoadingHair,
            queue: Array.from(this.loadingQueue),
            hasLoadedDefault: this.hasLoadedDefault
        };
    }

    // Method to manually trigger default loading
    async requestDefaultAvatar() {
        if (!this.hasLoadedDefault && !this.isLoadingAvatar) {
            await this.loadDefaultAvatarManually();
        } else {
            console.log('⚠️ Default avatar already loaded or request in progress');
        }
    }
}

// Make globally available
window.CustomizableGLBAvatarManager = CustomizableGLBAvatarManager;
window.GLBAvatarManager = CustomizableGLBAvatarManager;

// FIXED: Simplified console commands using your exact coordinates
window.testPerfectPositioning = function(height = 'medium') {
    if (!window.avatarManager || !window.avatarManager.currentHairModel) {
        console.log('❌ Hair model not available');
        return false;
    }

    const coords = {
        'short': { x: 0, y: -0.534, z: -0.13, scale: 1.0 },
        'medium': { x: 0, y: -0.39, z: -0.12, scale: 1.0 },
        'long': { x: 0, y: -0.534, z: -0.06, scale: 1.0 }
    };

    const targetCoords = coords[height];
    if (!targetCoords) {
        console.log(`❌ Unknown height: ${height}`);
        return false;
    }

    const hair = window.avatarManager.currentHairModel;
    hair.position.set(targetCoords.x, targetCoords.y, targetCoords.z);
    hair.scale.setScalar(targetCoords.scale);

    console.log(`✅ Applied perfect positioning for ${height} avatar`);
    console.log(`📍 Position: (${targetCoords.x}, ${targetCoords.y}, ${targetCoords.z}), Scale: ${targetCoords.scale}x`);
    return true;
};

window.switchToShort = async function() {
    if (window.avatarManager) {
        await window.avatarManager.setConfiguration({ height: 'short' });
        console.log('🔄 Switched to short avatar - hair will auto-position');
    }
};

window.switchToMedium = async function() {
    if (window.avatarManager) {
        await window.avatarManager.setConfiguration({ height: 'medium' });
        console.log('🔄 Switched to medium avatar - hair will auto-position');
    }
};

window.switchToLong = async function() {
    if (window.avatarManager) {
        await window.avatarManager.setConfiguration({ height: 'long' });
        console.log('🔄 Switched to long avatar - hair will auto-position');
    }
};
