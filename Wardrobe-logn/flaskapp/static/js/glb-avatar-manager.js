/**
 * COMPLETELY FIXED: Enhanced GLB Avatar Manager - NO CONTINUOUS LOADING
 * This version completely eliminates the continuous loading issue
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

        // FIXED: Add comprehensive loading states
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
            hairType: 'short_messy',
            hairColor: 'brown',
            eyeColor: 'brown'
        };

        // Hair styles configuration
        this.hairStyles = {
            female: {
                'afro_ponytail': {
                    name: 'Afro Ponytail',
                    preview: '/static/models/makehuman/hair/previews/afro_ponytail.jpg',
                    glbPath: '/static/models/makehuman/hair/Afro_Hair_-_Ponytail/hair_08.glb',
                    category: 'long'
                },
                'elvis_hazel': {
                    name: 'Elvis Hazel',
                    preview: '/static/models/makehuman/hair/previews/elvis_hazel.jpg',
                    glbPath: '/static/models/makehuman/hair/Elvs_Hazel_Hair/hair_08.glb',
                    category: 'medium'
                },
                'french_bob': {
                    name: 'French Bob',
                    preview: '/static/models/makehuman/hair/previews/french_bob.jpg',
                    glbPath: '/static/models/makehuman/hair/French_Bob_Blonde/hair_08.glb',
                    category: 'short'
                },
                'hair_06': {
                    name: 'Classic Style',
                    preview: '/static/models/makehuman/hair/previews/hair_06.jpg',
                    glbPath: '/static/models/makehuman/hair/Hair_06/hair_08.glb',
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
                'female_medium': {
                    name: 'Medium Length',
                    preview: '/static/models/makehuman/hair/previews/hair_female_medium.jpg',
                    glbPath: '/static/models/makehuman/hair/hair_female_medium/hair_08.glb',
                    category: 'medium'
                },
                'hair1': {
                    name: 'Short Curly',
                    preview: '/static/models/makehuman/hair/previews/hair1.jpg',
                    glbPath: '/static/models/makehuman/hair/hair1/hair_08.glb',
                    category: 'short'
                },
                'long_alpha': {
                    name: 'Long Alpha',
                    preview: '/static/models/makehuman/hair/previews/hairstyle_long2_alpha.jpg',
                    glbPath: '/static/models/makehuman/hair/Hairstyle_Long2_alpha_7_adaptation/hair_08.glb',
                    category: 'long'
                },
                'helen_troy': {
                    name: 'Helen of Troy',
                    preview: '/static/models/makehuman/hair/previews/helen_of_troy.jpg',
                    glbPath: '/static/models/makehuman/hair/Helen_Of_Troy/hair_08.glb',
                    category: 'long'
                },
                'shaggy_green': {
                    name: 'Shaggy Style',
                    preview: '/static/models/makehuman/hair/previews/shaggy_green.jpg',
                    glbPath: '/static/models/makehuman/hair/Shaggy_Green_Hair/hair_08.glb',
                    category: 'medium'
                },
                'short_messy': {
                    name: 'Short Messy',
                    preview: '/static/models/makehuman/hair/previews/short_messy.jpg',
                    glbPath: '/static/models/makehuman/hair/Short_Messy_Hair/hair_08.glb',
                    category: 'short'
                },
                'southern_belle': {
                    name: 'Southern Belle',
                    preview: '/static/models/makehuman/hair/previews/southern_belle.jpg',
                    glbPath: '/static/models/makehuman/hair/Southern_Belle_Ringlets/hair_08.glb',
                    category: 'long'
                },
                'bald': {
                    name: 'Bald',
                    preview: '/static/models/makehuman/hair/previews/bald.jpg',
                    glbPath: null,
                    category: 'none'
                }
            },
            male: {
                'male_short': {
                    name: 'Short Male',
                    preview: '/static/models/makehuman/hair/previews/male_short.jpg',
                    glbPath: '/static/models/makehuman/hair/male_short_hair/hair_08.glb',
                    category: 'short'
                },
                'minoan_hairdo': {
                    name: 'Minoan Style',
                    preview: '/static/models/makehuman/hair/previews/minoan_hairdo.jpg',
                    glbPath: '/static/models/makehuman/hair/Minoan_Hairdo_One/hair_08.glb',
                    category: 'medium'
                },
                'crew_cut': {
                    name: 'Crew Cut',
                    preview: '/static/models/makehuman/hair/previews/crew_cut.jpg',
                    glbPath: '/static/models/makehuman/hair/Crew_Cut/hair_08.glb',
                    category: 'short'
                },
                'business_cut': {
                    name: 'Business Cut',
                    preview: '/static/models/makehuman/hair/previews/business_cut.jpg',
                    glbPath: '/static/models/makehuman/hair/Business_Cut/hair_08.glb',
                    category: 'short'
                },
                'bald': {
                    name: 'Bald',
                    preview: '/static/models/makehuman/hair/previews/bald.jpg',
                    glbPath: null,
                    category: 'none'
                }
            }
        };

        // Eye texture mapping
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

        // Color definitions
        this.skinColors = {
            'light': 0xfdbcb4,
            'medium': 0xee9b82,
            'tan': 0xd08b5b,
            'dark': 0xae5d29,
            'darker': 0x8b4513,
            'darkest': 0x654321,
            'pink': 0xffb6c1,
            'olive': 0xddbea9
        };

        this.eyeColors = {
            'brown': 0x6B4423,
            'blue': 0x4A90E2,
            'green': 0x50C878,
            'light-blue': 0x87CEEB,
            'purple': 0x800080,
            'gray': 0x708090,
            'grey': 0x708090,
            'dark-green': 0x228B22,
            'hazel': 0xB8860B
        };

        this.hairColors = {
            'brown': 0x8B4513,
            'black': 0x000000,
            'blonde': 0xFFD700,
            'red': 0xDC143C,
            'auburn': 0xA0522D,
            'gray': 0xC0C0C0,
            'white': 0xFFFFFF,
            'pink': 0xFF69B4
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

        // SMART Hair Positioning System
        this.smartHairPositioning = new SmartHairPositioningSystem(this);

        // FIXED: Only initialize once, don't load default avatar automatically
        this.init();
    }

    init() {
        console.log('🤖 Initializing FIXED GLB Avatar Manager (NO continuous loading)...');
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.setupControls();
        this.setupLoaders();
        this.setupGround();
        this.animate();

        // FIXED: Set initialized flag but DON'T load default avatar automatically
        this.isInitialized = true;
        console.log('✅ FIXED GLB Avatar Manager initialized - waiting for manual load request');
    }

    // FIXED: Manual method to load default avatar (called only when needed)
    async loadDefaultAvatarManually() {
        if (this.hasLoadedDefault || this.isLoadingAvatar) {
            console.log('⚠️ Default avatar already loaded or loading');
            return;
        }

        console.log('🤖 Loading default GLB avatar (MANUAL REQUEST)...');
        this.hasLoadedDefault = true;

        try {
            await this.loadAvatarFromConfig(this.config);
            console.log('✅ Default avatar loaded successfully');

            // Load default hair after avatar loads (ONE-TIME only)
            setTimeout(async () => {
                if (!this.isLoadingHair && !this.currentHairModel) {
                    await this.updateHairStyle(this.config.hairType);
                }
            }, 1000);
        } catch (error) {
            console.error('❌ Failed to load default avatar:', error);
            this.hasLoadedDefault = false; // Allow retry
            this.loadFallbackAvatar();
        }
    }

    // FIXED: Prevent duplicate loading
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

    // FIXED: Comprehensive loading protection
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

        // FIXED: Set all protection flags
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

            console.log('✅ GLB avatar loaded - NO MORE CONTINUOUS LOADING');
            return this.avatarModel;

        } catch (error) {
            console.error('❌ Error loading GLB avatar:', error);
            throw error;
        } finally {
            // FIXED: Always clear loading flags
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

    // FIXED: Hair loading with protection
    async updateHairStyle(hairStyleKey) {
        if (this.isLoadingHair) {
            console.log('⚠️ Hair loading blocked - already in progress');
            return false;
        }

        console.log(`🦱 Updating hair style to: ${hairStyleKey}`);

        const gender = this.config.gender || 'female';
        const hairData = this.hairStyles[gender][hairStyleKey];

        if (!hairData) {
            console.error(`❌ Hair style not found: ${hairStyleKey} for gender: ${gender}`);
            return false;
        }

        // FIXED: Set loading protection
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
                // Apply positioning
                const foreheadResult = this.positionHairOnForehead();

                if (!foreheadResult) {
                    console.warn('⚠️ Forehead positioning failed, using smart positioning');
                    const result = this.smartHairPositioning.positionHairCloseToHead(hairModel, this.avatarModel);

                    if (result.success) {
                        console.log(`🧠 SMART positioning successful`);
                    } else {
                        console.warn('⚠️ Smart positioning failed, using fallback');
                        this.positionHairOnAvatarFallback(hairModel);
                    }
                }

                // Apply materials and add to scene
                this.enhanceHairMaterials(hairModel);
                this.scene.add(hairModel);
                this.currentHairModel = hairModel;
                this.currentHairModel.name = `hair_${hairStyleKey}`;
                this.config.hairType = hairStyleKey;

                console.log(`✅ GLB hair model loaded: ${hairData.name}`);
                return true;
            }
        } catch (error) {
            console.error(`❌ Failed to load GLB hair model: ${error.message}`);
            return false;
        } finally {
            // FIXED: Always clear hair loading flag
            this.isLoadingHair = false;
            console.log('🔓 Hair loading flag cleared');
        }

        return false;
    }

    // Include all the setup methods
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

    // Forehead positioning methods
    moveHairToForehead() {
        if (!this.currentHairModel || !this.avatarModel) {
            console.log('❌ No avatar or hair model available');
            return false;
        }

        const hairModel = this.currentHairModel;
        const avatarModel = this.avatarModel;

        console.log('🎯 Moving hair to optimal forehead position...');

        const avatarBox = new THREE.Box3().setFromObject(avatarModel);
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());
        const avatarSize = avatarBox.getSize(new THREE.Vector3());

        const hairY = avatarBox.max.y - (avatarSize.y * 0.082);
        const hairZ = avatarCenter.z - (avatarSize.z * 0.26);
        const hairX = avatarCenter.x;

        hairModel.position.set(hairX, hairY, hairZ);

        const newHairBox = new THREE.Box3().setFromObject(hairModel);
        const hairBottom = newHairBox.min.y;
        const yAdjustment = hairY - hairBottom;
        hairModel.position.y += yAdjustment;

        console.log(`✅ Hair positioned at forehead: (${hairModel.position.x.toFixed(3)}, ${hairModel.position.y.toFixed(3)}, ${hairModel.position.z.toFixed(3)})`);
        return true;
    }

    positionHairOnForehead() {
        console.log('🎯 Starting forehead hair positioning...');

        const foreheadPos = this.createForeheadReference();
        if (!foreheadPos) return false;

        if (!this.moveHairToForehead()) return false;

        this.fineAdjustHairPosition(0, -0.01, 0.02);
        this.optimizeHairScale(0.95);

        console.log('✅ Forehead positioning complete!');
        return true;
    }

    createForeheadReference() {
        if (!this.avatarModel) {
            console.log('❌ No avatar model available');
            return false;
        }

        const existingMarker = this.scene.getObjectByName('forehead_reference');
        if (existingMarker) {
            this.scene.remove(existingMarker);
        }

        const avatarBox = new THREE.Box3().setFromObject(this.avatarModel);
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());
        const avatarSize = avatarBox.getSize(new THREE.Vector3());

        const foreheadPos = new THREE.Vector3(
            avatarCenter.x,
            avatarBox.max.y - (avatarSize.y * 0.082),
            avatarCenter.z - (avatarSize.z * 0.26)
        );

        const markerGeometry = new THREE.SphereGeometry(0.02, 8, 8);
        const markerMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.8
        });

        const foreheadMarker = new THREE.Mesh(markerGeometry, markerMaterial);
        foreheadMarker.position.copy(foreheadPos);
        foreheadMarker.name = 'forehead_reference';

        this.scene.add(foreheadMarker);

        console.log('🎯 Cyan forehead reference marker created');
        return foreheadPos;
    }

    fineAdjustHairPosition(xOffset = 0, yOffset = 0, zOffset = 0) {
        if (!this.currentHairModel) {
            console.log('❌ No hair model available');
            return false;
        }

        const hairModel = this.currentHairModel;
        hairModel.position.x += xOffset;
        hairModel.position.y += yOffset;
        hairModel.position.z += zOffset;

        console.log(`🔧 Hair position adjusted by (${xOffset}, ${yOffset}, ${zOffset})`);
        return true;
    }

    optimizeHairScale(scaleFactor = 0.9) {
        if (!this.currentHairModel) {
            console.log('❌ No hair model available');
            return false;
        }

        const hairModel = this.currentHairModel;
        const currentScale = hairModel.scale.x;
        const newScale = currentScale * scaleFactor;

        hairModel.scale.setScalar(newScale);

        console.log(`📏 Hair scale adjusted: ${currentScale.toFixed(3)} → ${newScale.toFixed(3)}`);
        return true;
    }

    // Continue with other essential methods...
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

    // Essential utility methods
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

    positionHairOnAvatarFallback(hairModel) {
        console.log('🔄 Using fallback hair positioning...');

        const avatarBox = new THREE.Box3().setFromObject(this.avatarModel);
        const avatarSize = avatarBox.getSize(new THREE.Vector3());
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());
        const avatarScale = this.avatarModel.scale.x;

        const hairScale = avatarScale * 0.85;
        hairModel.scale.setScalar(hairScale);

        hairModel.position.copy(avatarCenter);
        hairModel.position.y = avatarBox.min.y + (avatarSize.y * 0.95);
        hairModel.rotation.set(0, 0, 0);

        console.log('✅ Fallback hair positioning complete');
    }

    // Update methods for UI
    async updateGender(gender) {
        console.log(`👤 Updating gender to: ${gender}`);
        this.config.gender = gender;
        this.removeCurrentHair();
        await this.loadAvatarFromConfig(this.config);

        const defaultHairStyles = {
            'female': 'short_messy',
            'male': 'male_short'
        };

        const defaultHair = defaultHairStyles[gender] || 'short_messy';
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

        console.log('🧹 FIXED Avatar Manager cleaned up');
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

    // FIXED: Method to manually trigger default loading (call this from UI)
    async requestDefaultAvatar() {
        if (!this.hasLoadedDefault && !this.isLoadingAvatar) {
            await this.loadDefaultAvatarManually();
        } else {
            console.log('⚠️ Default avatar already loaded or request in progress');
        }
    }

    enableHairDebugMode(enabled = true) {
        this.smartHairPositioning.setDebugMode(enabled);
    }

    getHairPositioningInfo() {
        return this.smartHairPositioning.getLastPositioningInfo();
    }
}

// Simplified Smart Hair Positioning System
class SmartHairPositioningSystem {
    constructor(avatarManager) {
        this.avatarManager = avatarManager;
        this.debugMode = false;
        this.lastPositioningInfo = null;
    }

    positionHairCloseToHead(hairModel, avatarModel) {
        console.log('🧠 SMART: Starting hair positioning close to head...');

        try {
            const headInfo = this.findHeadPosition(avatarModel);
            const hairAnalysis = this.analyzeHairModel(hairModel);
            const positioningResult = this.calculateOptimalPositioning(headInfo, hairAnalysis, avatarModel);

            this.applyPositioning(hairModel, positioningResult);

            this.lastPositioningInfo = {
                headInfo,
                hairAnalysis,
                positioningResult,
                timestamp: Date.now()
            };

            return {
                success: true,
                method: headInfo.method,
                confidence: headInfo.confidence
            };
        } catch (error) {
            console.error('❌ SMART positioning failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    findHeadPosition(avatarModel) {
        const avatarBox = new THREE.Box3().setFromObject(avatarModel);
        const avatarSize = avatarBox.getSize(new THREE.Vector3());
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());

        const headY = avatarBox.min.y + (avatarSize.y * 0.88);

        return {
            position: new THREE.Vector3(avatarCenter.x, headY, avatarCenter.z),
            method: 'geometric',
            confidence: 0.7
        };
    }

    analyzeHairModel(hairModel) {
        const hairBox = new THREE.Box3().setFromObject(hairModel);
        const hairSize = hairBox.getSize(new THREE.Vector3());

        return {
            size: hairSize,
            type: 'medium'
        };
    }

    calculateOptimalPositioning(headInfo, hairAnalysis, avatarModel) {
        const avatarBox = new THREE.Box3().setFromObject(avatarModel);
        const avatarSize = avatarBox.getSize(new THREE.Vector3());
        const avatarScale = avatarModel.scale.x || 1.0;

        const estimatedHeadWidth = avatarSize.x * 0.22;
        const baseScale = estimatedHeadWidth / hairAnalysis.size.x;
        const finalScale = Math.max(0.4, Math.min(2.5, baseScale * 1.1 * avatarScale));

        const targetPosition = headInfo.position.clone();
        targetPosition.y -= 0.06 * avatarSize.y;

        return {
            scale: finalScale,
            position: targetPosition
        };
    }

    applyPositioning(hairModel, positioningResult) {
        hairModel.scale.setScalar(positioningResult.scale);
        hairModel.position.copy(positioningResult.position);

        const hairBox = new THREE.Box3().setFromObject(hairModel);
        const yAdjustment = positioningResult.position.y - hairBox.min.y;
        hairModel.position.y += yAdjustment - 0.02;

        hairModel.rotation.set(0, 0, 0);
    }

    setDebugMode(enabled) {
        this.debugMode = enabled;
    }

    getLastPositioningInfo() {
        return this.lastPositioningInfo;
    }
}

// Make globally available
window.CustomizableGLBAvatarManager = CustomizableGLBAvatarManager;
window.GLBAvatarManager = CustomizableGLBAvatarManager; // Alias
window.SmartHairPositioningSystem = SmartHairPositioningSystem;

console.log('✅ COMPLETELY FIXED: GLB Avatar Manager - NO CONTINUOUS LOADING');
console.log('🔒 Loading protection: Multiple flags prevent duplicate requests');
console.log('📞 Call avatarManager.requestDefaultAvatar() to load default avatar manually');
console.log('📊 Use avatarManager.getLoadingStatus() to check current state');