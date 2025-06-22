// ENHANCED AVATAR CLOTHING RENDERER WITH VERTEX COLORING SYSTEM
// Integrates the sophisticated vertex coloring from wardrobe app into avatar system

class AvatarVertexColoringRenderer {
    constructor() {
        console.log('🎨 Initializing Avatar Vertex Coloring Renderer...');

        this.scene = null;
        this.avatar = null;
        this.currentClothing = new Map();
        this.objLoader = null;
        this.clothingPositions = new Map();
        this.clothingScales = new Map();

        this.initializeVertexColorOBJLoader();
        console.log('✅ Avatar Vertex Coloring Renderer initialized');
    }

    initializeVertexColorOBJLoader() {
        try {
            if (typeof THREE === 'undefined') {
                throw new Error('THREE.js not loaded');
            }

            // Initialize the enhanced OBJ loader with vertex color support
            this.objLoader = new EnhancedVertexColorOBJLoader();
            console.log('🎨 Vertex Color OBJ Loader initialized for avatar system');
        } catch (error) {
            console.error('❌ Failed to initialize vertex color loader:', error);
        }
    }

    setReferences(scene, avatar) {
        console.log('🔗 Setting avatar vertex color renderer references...');
        this.scene = scene;
        this.avatar = avatar;

        if (this.scene && this.avatar) {
            console.log('✅ Avatar vertex color renderer references set successfully');
            return true;
        } else {
            console.warn('⚠️ Invalid references provided to avatar vertex color renderer');
            return false;
        }
    }

    // PRIORITY LOADING SYSTEM WITH VERTEX COLORING
    async loadClothingFromDatabase(itemId) {
        console.log(`🎨 Loading clothing with vertex coloring: ${itemId}`);

        if (!this.scene || !this.avatar) {
            console.error('❌ Scene or avatar not set');
            throw new Error('Scene or avatar not set in avatar vertex renderer');
        }

        try {
            // Fetch item data
            const response = await fetch(`/api/wardrobe/item/${itemId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const itemData = await response.json();
            console.log('📊 Avatar item data received:', itemData);

            if (!itemData.success) {
                throw new Error(itemData.error || 'Failed to fetch item data');
            }

            // PRIORITY 1: Load vertex-colored OBJ files
            if (itemData.has_3d_model && itemData.model_3d_path) {
                console.log(`🎨 Found saved vertex-colored model: ${itemData.model_3d_path}`);

                try {
                    const checkResponse = await fetch(itemData.model_3d_path, { method: 'HEAD' });
                    if (checkResponse.ok) {
                        console.log(`🎨 Loading vertex-colored OBJ: ${itemData.model_3d_path}`);
                        return await this.loadVertexColoredOBJ(itemData.model_3d_path, itemData);
                    }
                } catch (e) {
                    console.warn(`⚠️ Saved model not accessible: ${e.message}`);
                }
            }

            // PRIORITY 2: Search for generated vertex-colored OBJ files
            const userId = itemData.userId || itemData.user_id;
            const modelTaskId = itemData.model_task_id || itemData.modelTaskId;

            if (modelTaskId && !modelTaskId.startsWith('auto_') && !modelTaskId.startsWith('fallback_')) {
                console.log(`🔍 Searching for vertex-colored OBJ with task_id: ${modelTaskId}`);

                const patterns = [
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_0.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_1.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_2.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_3.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}_4.obj`,
                    `/static/models/generated/${userId}/colab_model_task_${modelTaskId}.obj`
                ];

                for (const pattern of patterns) {
                    try {
                        const checkResponse = await fetch(pattern, { method: 'HEAD' });
                        if (checkResponse.ok) {
                            console.log(`🎨 Found vertex-colored OBJ: ${pattern}`);
                            return await this.loadVertexColoredOBJ(pattern, itemData);
                        }
                    } catch (e) {
                        continue;
                    }
                }
            }

            // PRIORITY 3: Create vertex-colored plane from image analysis
            if (itemData.file_path || itemData.texture_preview_path) {
                console.log(`🎨 Creating vertex-colored plane from image analysis...`);
                return await this.createVertexColoredPlaneFromImage(itemData);
            }

            // PRIORITY 4: Create vertex-colored fallback based on color data
            console.warn(`⚠️ No OBJ found, creating vertex-colored fallback...`);
            return await this.createVertexColoredFallback(itemData);

        } catch (error) {
            console.error('❌ Avatar vertex coloring failed:', error);
            return await this.createVertexColoredFallback(itemData);
        }
    }

    // LOAD VERTEX-COLORED OBJ FILES
    async loadVertexColoredOBJ(objPath, itemData) {
        return new Promise((resolve, reject) => {
            console.log(`🎨 Loading vertex-colored OBJ for avatar: ${objPath}`);

            this.objLoader.load(
                objPath,
                (object) => {
                    console.log('✅ Vertex-colored OBJ loaded for avatar, processing...');
                    this.processVertexColoredObjectForAvatar(object, itemData);
                    resolve(true);
                },
                (progress) => {
                    if (progress.lengthComputable) {
                        const percent = (progress.loaded / progress.total * 100).toFixed(1);
                        console.log(`📊 Avatar loading progress: ${percent}%`);
                    }
                },
                (error) => {
                    console.error('❌ Avatar vertex-colored OBJ loading failed:', error);
                    reject(error);
                }
            );
        });
    }

    // PROCESS VERTEX-COLORED OBJECT FOR AVATAR SYSTEM
    processVertexColoredObjectForAvatar(object, itemData) {
        console.log('🎨 Processing vertex-colored object for avatar positioning...');

        // Apply avatar-specific rotation (consistent with avatar system)
        this.applyAvatarClothingRotation(object, itemData);

        // Position on avatar body
        const clothingType = this.determineAvatarClothingType(itemData);
        this.positionClothingOnAvatar(object, clothingType);

        // Add to avatar scene
        this.scene.add(object);

        // Store in avatar clothing system
        const clothingData = {
            mesh: object,
            itemData: itemData,
            type: clothingType,
            category: itemData.type || clothingType,
            source: 'vertex_colored_obj',
            hasVertexColors: true
        };

        this.currentClothing.set(itemData._id, clothingData);
        console.log(`✅ Vertex-colored ${clothingType} added to avatar`);
    }

    // CREATE VERTEX-COLORED PLANE FROM IMAGE ANALYSIS
    async createVertexColoredPlaneFromImage(itemData) {
        console.log('🎨 Creating vertex-colored plane for avatar from image analysis...');

        try {
            const clothingType = this.determineAvatarClothingType(itemData);

            // Analyze image for color palette
            const colorPalette = await this.analyzeImageForVertexColors(itemData.file_path || itemData.texture_preview_path);

            // Create geometry with vertex colors
            const geometry = this.createVertexColoredAvatarGeometry(clothingType, colorPalette);

            // Create material for vertex coloring
            const material = new THREE.MeshLambertMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                transparent: false
            });

            const mesh = new THREE.Mesh(geometry, material);

            // Apply avatar positioning
            this.applyAvatarClothingRotation(mesh, itemData);
            this.positionClothingOnAvatar(mesh, clothingType);

            this.scene.add(mesh);

            const clothingData = {
                mesh: mesh,
                itemData: itemData,
                type: clothingType,
                category: itemData.type || clothingType,
                source: 'vertex_colored_plane',
                hasVertexColors: true
            };

            this.currentClothing.set(itemData._id, clothingData);
            console.log(`✅ Vertex-colored plane ${clothingType} created for avatar`);
            return true;

        } catch (error) {
            console.error('❌ Vertex-colored plane creation failed:', error);
            return await this.createVertexColoredFallback(itemData);
        }
    }

    // ANALYZE IMAGE FOR VERTEX COLOR PALETTE
    async analyzeImageForVertexColors(imagePath) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";

            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;

                ctx.drawImage(img, 0, 0);

                // Sample colors from different regions
                const palette = [];
                const samplePoints = [
                    { x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, // Top corners
                    { x: 0.5, y: 0.5 }, // Center
                    { x: 0.2, y: 0.8 }, { x: 0.8, y: 0.8 }, // Bottom corners
                    { x: 0.1, y: 0.5 }, { x: 0.9, y: 0.5 }, // Side centers
                    { x: 0.5, y: 0.1 }, { x: 0.5, y: 0.9 }  // Top/bottom centers
                ];

                samplePoints.forEach(point => {
                    const x = Math.floor(point.x * canvas.width);
                    const y = Math.floor(point.y * canvas.height);
                    const imageData = ctx.getImageData(x, y, 1, 1);
                    const [r, g, b] = imageData.data;

                    palette.push({
                        r: r / 255,
                        g: g / 255,
                        b: b / 255
                    });
                });

                console.log(`🎨 Extracted ${palette.length} colors for vertex coloring`);
                resolve(palette);
            };

            img.onerror = () => {
                console.warn('⚠️ Image analysis failed, using default palette');
                resolve(this.getDefaultVertexColorPalette());
            };

            img.src = imagePath;
        });
    }

    // CREATE VERTEX-COLORED GEOMETRY FOR AVATAR
    createVertexColoredAvatarGeometry(clothingType, colorPalette) {
        let geometry;

        // Create more detailed geometry for better vertex coloring
        switch (clothingType) {
            case 'top':
                geometry = new THREE.PlaneGeometry(1.2, 1.0, 8, 6); // 8x6 segments for vertex colors
                break;
            case 'bottom':
                geometry = new THREE.PlaneGeometry(1.0, 1.2, 6, 8); // 6x8 segments
                break;
            case 'dress':
                geometry = new THREE.PlaneGeometry(1.2, 1.8, 8, 12); // 8x12 segments for detailed coloring
                break;
            case 'outerwear':
                geometry = new THREE.PlaneGeometry(1.3, 1.1, 9, 7); // 9x7 segments
                break;
            case 'shoes':
                geometry = new THREE.PlaneGeometry(0.6, 0.8, 4, 5); // 4x5 segments
                break;
            default:
                geometry = new THREE.PlaneGeometry(1.0, 1.0, 6, 6); // 6x6 segments
        }

        // Apply vertex colors based on palette
        this.applyVertexColorsToGeometry(geometry, colorPalette);

        return geometry;
    }

    // APPLY VERTEX COLORS TO GEOMETRY
    applyVertexColorsToGeometry(geometry, colorPalette) {
        const colors = [];
        const positionAttribute = geometry.attributes.position;
        const vertexCount = positionAttribute.count;

        console.log(`🎨 Applying vertex colors to ${vertexCount} vertices`);

        for (let i = 0; i < vertexCount; i++) {
            // Create color variation based on vertex position
            const colorIndex = Math.floor((i / vertexCount) * colorPalette.length);
            const baseColor = colorPalette[colorIndex] || colorPalette[0];

            // Add slight variation for natural look
            const variation = 0.1;
            const r = Math.max(0, Math.min(1, baseColor.r + (Math.random() - 0.5) * variation));
            const g = Math.max(0, Math.min(1, baseColor.g + (Math.random() - 0.5) * variation));
            const b = Math.max(0, Math.min(1, baseColor.b + (Math.random() - 0.5) * variation));

            colors.push(r, g, b);
        }

        // Set vertex colors
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        console.log(`✅ Applied vertex colors to avatar clothing geometry`);
    }

    // CREATE VERTEX-COLORED FALLBACK
    async createVertexColoredFallback(itemData) {
        console.log('🎨 Creating vertex-colored fallback for avatar...');

        try {
            const clothingType = this.determineAvatarClothingType(itemData);

            // Extract colors from database
            const colorPalette = this.extractColorsFromItemData(itemData);

            // Create vertex-colored geometry
            const geometry = this.createVertexColoredAvatarGeometry(clothingType, colorPalette);

            // Material for vertex coloring
            const material = new THREE.MeshLambertMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                roughness: 0.8 // Fabric-like appearance
            });

            const mesh = new THREE.Mesh(geometry, material);

            // Apply avatar positioning
            this.applyAvatarClothingRotation(mesh, itemData);
            this.positionClothingOnAvatar(mesh, clothingType);

            this.scene.add(mesh);

            const clothingData = {
                mesh: mesh,
                itemData: itemData,
                type: clothingType,
                category: itemData.type || clothingType,
                source: 'vertex_colored_fallback',
                hasVertexColors: true
            };

            this.currentClothing.set(itemData._id, clothingData);
            console.log(`✅ Vertex-colored fallback ${clothingType} created for avatar`);
            return true;

        } catch (error) {
            console.error('❌ Vertex-colored fallback failed:', error);
            return false;
        }
    }

    // EXTRACT COLORS FROM ITEM DATA FOR VERTEX COLORING
    extractColorsFromItemData(itemData) {
        let basePalette = this.getDefaultVertexColorPalette();

        if (itemData.color) {
            let primaryColor = { r: 0.8, g: 0.8, b: 0.8 };

            if (typeof itemData.color === 'object' && itemData.color.rgb) {
                const [r, g, b] = itemData.color.rgb;
                primaryColor = { r: r / 255, g: g / 255, b: b / 255 };
            } else if (typeof itemData.color === 'string' && itemData.color.includes(' ')) {
                const colorValues = itemData.color.split(' ').map(v => parseInt(v.trim()));
                if (colorValues.length >= 3) {
                    const [r, g, b] = colorValues;
                    primaryColor = { r: r / 255, g: g / 255, b: b / 255 };
                }
            }

            // Create palette variations around primary color
            basePalette = this.generateVertexColorPalette(primaryColor);
        }

        return basePalette;
    }

    // GENERATE VERTEX COLOR PALETTE
    generateVertexColorPalette(primaryColor) {
        const palette = [];

        // Add primary color
        palette.push(primaryColor);

        // Add lighter variations
        palette.push({
            r: Math.min(1, primaryColor.r * 1.2),
            g: Math.min(1, primaryColor.g * 1.2),
            b: Math.min(1, primaryColor.b * 1.2)
        });

        // Add darker variations
        palette.push({
            r: primaryColor.r * 0.8,
            g: primaryColor.g * 0.8,
            b: primaryColor.b * 0.8
        });

        // Add complementary tones
        palette.push({
            r: Math.min(1, primaryColor.r * 0.9 + 0.1),
            g: Math.min(1, primaryColor.g * 0.9 + 0.1),
            b: Math.min(1, primaryColor.b * 0.9 + 0.1)
        });

        return palette;
    }

    // GET DEFAULT VERTEX COLOR PALETTE
    getDefaultVertexColorPalette() {
        return [
            { r: 0.8, g: 0.8, b: 0.8 }, // Light gray
            { r: 0.6, g: 0.6, b: 0.6 }, // Medium gray
            { r: 0.4, g: 0.4, b: 0.4 }, // Dark gray
            { r: 0.9, g: 0.9, b: 0.9 }  // Very light gray
        ];
    }

    // AVATAR-SPECIFIC CLOTHING ROTATION
    applyAvatarClothingRotation(object, itemData) {
        const clothingType = this.determineAvatarClothingType(itemData);
        console.log(`🔄 Applying avatar clothing rotation for ${clothingType}...`);

        // Reset rotations
        object.rotation.set(0, 0, 0);

        // Apply corrective rotations (consistent with avatar system)
        switch (clothingType) {
            case 'top':
            case 'bottom':
            case 'dress':
            case 'outerwear':
            case 'shoes':
                object.rotation.set(-Math.PI / 2, 0, Math.PI / 2);
                break;
            default:
                object.rotation.set(-Math.PI / 2, 0, Math.PI / 2);
        }

        console.log(`✅ Avatar clothing rotation applied: x=${object.rotation.x.toFixed(2)}, y=${object.rotation.y.toFixed(2)}, z=${object.rotation.z.toFixed(2)}`);
    }

    // DETERMINE AVATAR CLOTHING TYPE
    determineAvatarClothingType(itemData) {
        const label = (itemData.label || itemData.type || '').toLowerCase();
        const category = (itemData.category || itemData.type || '').toLowerCase();

        if (label.includes('shirt') || label.includes('top') || label.includes('pullover') || category === 'tops') {
            return 'top';
        } else if (label.includes('trouser') || label.includes('pant') || category === 'bottoms') {
            return 'bottom';
        } else if (label.includes('dress') || category === 'dresses') {
            return 'dress';
        } else if (label.includes('coat') || label.includes('jacket') || category === 'outerwear') {
            return 'outerwear';
        } else if (label.includes('shoe') || label.includes('sandal') || label.includes('boot') || category === 'shoes') {
            return 'shoes';
        } else {
            return 'top';
        }
    }

    // POSITION CLOTHING ON AVATAR (INHERITED FROM AVATAR SYSTEM)
    positionClothingOnAvatar(mesh, clothingType) {
        if (!this.avatar) {
            console.warn('⚠️ No avatar available for positioning');
            return;
        }

        const avatarBox = new THREE.Box3().setFromObject(this.avatar);
        const avatarHeight = avatarBox.max.y - avatarBox.min.y;
        const avatarCenter = avatarBox.getCenter(new THREE.Vector3());

        switch (clothingType) {
            case 'top':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y + avatarHeight * 0.15,
                    avatarCenter.z + 0.1
                );
                break;
            case 'bottom':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y - avatarHeight * 0.15,
                    avatarCenter.z + 0.1
                );
                break;
            case 'dress':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y - avatarHeight * 0.05,
                    avatarCenter.z + 0.1
                );
                break;
            case 'outerwear':
                mesh.position.set(
                    avatarCenter.x,
                    avatarCenter.y + avatarHeight * 0.15,
                    avatarCenter.z + 0.15
                );
                break;
            case 'shoes':
                mesh.position.set(
                    avatarCenter.x,
                    avatarBox.min.y + 0.05,
                    avatarCenter.z + 0.05
                );
                break;
            default:
                mesh.position.copy(avatarCenter);
                mesh.position.z += 0.1;
        }

        // Avatar-proportional scaling
        let scale;
        switch (clothingType) {
            case 'dress':
                scale = avatarHeight / 2.2;
                break;
            case 'top':
                scale = avatarHeight / 2.8;
                break;
            case 'bottom':
                scale = avatarHeight / 3.0;
                break;
            default:
                scale = avatarHeight / 2.5;
        }

        mesh.scale.setScalar(scale);
        console.log(`📏 Avatar clothing positioned and scaled: ${clothingType}`);
    }

    // REMOVE CLOTHING (INHERITED FROM AVATAR SYSTEM)
    async removeClothing(itemId) {
        console.log(`🗑️ Removing vertex-colored clothing: ${itemId}`);

        if (!this.currentClothing.has(itemId)) {
            console.warn(`⚠️ Clothing ${itemId} not found`);
            return false;
        }

        try {
            const clothingData = this.currentClothing.get(itemId);

            if (clothingData.mesh && this.scene) {
                this.scene.remove(clothingData.mesh);

                if (clothingData.mesh.geometry) {
                    clothingData.mesh.geometry.dispose();
                }

                if (clothingData.mesh.material) {
                    if (Array.isArray(clothingData.mesh.material)) {
                        clothingData.mesh.material.forEach(mat => mat.dispose());
                    } else {
                        clothingData.mesh.material.dispose();
                    }
                }
            }

            this.currentClothing.delete(itemId);
            this.clothingPositions.delete(itemId);
            this.clothingScales.delete(itemId);

            console.log(`✅ Vertex-colored clothing ${itemId} removed successfully`);
            return true;

        } catch (error) {
            console.error('❌ Error removing vertex-colored clothing:', error);
            return false;
        }
    }

    // CLEAR ALL CLOTHING
    async clearAllClothing() {
        console.log('🧹 Clearing all vertex-colored clothing...');

        const itemIds = Array.from(this.currentClothing.keys());
        let removedCount = 0;

        for (const itemId of itemIds) {
            const success = await this.removeClothing(itemId);
            if (success) {
                removedCount++;
            }
        }

        this.clothingPositions.clear();
        this.clothingScales.clear();

        console.log(`✅ Cleared ${removedCount} vertex-colored clothing items`);
        return removedCount;
    }

    // GET ACTIVE CLOTHING INFO
    getActiveClothing() {
        const activeItems = [];

        for (const [itemId, clothingData] of this.currentClothing.entries()) {
            activeItems.push({
                id: itemId,
                name: clothingData.itemData?.label || 'Unknown Item',
                type: clothingData.type,
                category: clothingData.category,
                source: clothingData.source,
                hasVertexColors: clothingData.hasVertexColors || false
            });
        }

        return activeItems;
    }
}

// ENHANCED VERTEX COLOR OBJ LOADER (FROM WARDROBE SYSTEM)
class EnhancedVertexColorOBJLoader {
    constructor(manager) {
        this.manager = manager || new THREE.LoadingManager();
    }

    load(url, onLoad, onProgress, onError) {
        const loader = new THREE.FileLoader(this.manager);
        loader.setResponseType('text');

        loader.load(url, (text) => {
            try {
                const object = this.parse(text);
                if (onLoad) onLoad(object);
            } catch (error) {
                console.error('🎨 Vertex Color OBJ Parse Error:', error);
                if (onError) onError(error);
            }
        }, onProgress, onError);
    }

    parse(text) {
        console.log('🎨 Parsing OBJ with vertex color support for avatar...');

        const vertices = [];
        const normals = [];
        const uvs = [];
        const vertexColors = [];
        const faces = [];

        const lines = text.split('\n');
        console.log(`📊 Processing ${lines.length} lines for avatar...`);

        let colorCount = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            if (line.length === 0 || line.charAt(0) === '#') continue;

            const parts = line.split(/\s+/);
            const type = parts[0];

            switch (type) {
                case 'v':
                    // Parse vertices: v x y z [r g b]
                    vertices.push(
                        parseFloat(parts[1]) || 0,
                        parseFloat(parts[2]) || 0,
                        parseFloat(parts[3]) || 0
                    );

                    // Check for vertex color data
                    if (parts.length >= 7) {
                        const r = parseFloat(parts[4]) || 0.8;
                        const g = parseFloat(parts[5]) || 0.8;
                        const b = parseFloat(parts[6]) || 0.8;

                        vertexColors.push(r, g, b);
                        colorCount++;

                        if (colorCount <= 5) {
                            console.log(`🎨 Avatar vertex ${Math.floor(vertices.length/3)}: color(${r.toFixed(3)}, ${g.toFixed(3)}, ${b.toFixed(3)})`);
                        }
                    } else {
                        vertexColors.push(0.8, 0.8, 0.8);
                    }
                    break;

                case 'vn':
                    normals.push(
                        parseFloat(parts[1]) || 0,
                        parseFloat(parts[2]) || 0,
                        parseFloat(parts[3]) || 0
                    );
                    break;

                case 'vt':
                    uvs.push(
                        parseFloat(parts[1]) || 0,
                        parseFloat(parts[2]) || 0
                    );
                    break;

                case 'f':
                    this.parseFace(parts.slice(1), faces);
                    break;
            }
        }

        console.log(`🎨 Avatar parsed: ${vertices.length/3} vertices, ${faces.length} faces`);
        console.log(`🎨 Found ${colorCount} vertices with color data for avatar`);

        return this.buildGeometry(vertices, normals, uvs, faces, vertexColors);
    }

    parseFace(faceData, faces) {
        const face = [];

        for (let i = 0; i < faceData.length; i++) {
            const vertex = faceData[i];
            const indices = vertex.split('/');

            face.push({
                vertex: indices[0] ? parseInt(indices[0]) - 1 : 0,
                uv: indices[1] ? parseInt(indices[1]) - 1 : null,
                normal: indices[2] ? parseInt(indices[2]) - 1 : null
            });
        }

        // Triangulate face
        for (let i = 1; i < face.length - 1; i++) {
            faces.push([face[0], face[i], face[i + 1]]);
        }
    }

    buildGeometry(vertices, normals, uvs, faces, vertexColors = []) {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const normalsArray = [];
        const uvsArray = [];
        const colorsArray = [];

        const hasColors = vertexColors.length > 0;
        console.log(`🎨 Building avatar geometry - Has colors: ${hasColors}, Color count: ${vertexColors.length/3}`);

        for (let i = 0; i < faces.length; i++) {
            const face = faces[i];

            for (let j = 0; j < 3; j++) {
                const vertexIndex = face[j].vertex;
                const normalIndex = face[j].normal;
                const uvIndex = face[j].uv;

                // Positions
                if (vertexIndex >= 0 && vertexIndex < vertices.length / 3) {
                    positions.push(
                        vertices[vertexIndex * 3] || 0,
                        vertices[vertexIndex * 3 + 1] || 0,
                        vertices[vertexIndex * 3 + 2] || 0
                    );
                } else {
                    positions.push(0, 0, 0);
                }

                // Normals
                if (normalIndex !== null && normalIndex >= 0 && normalIndex < normals.length / 3) {
                    normalsArray.push(
                        normals[normalIndex * 3] || 0,
                        normals[normalIndex * 3 + 1] || 1,
                        normals[normalIndex * 3 + 2] || 0
                    );
                } else {
                    normalsArray.push(0, 1, 0);
                }

                // UVs
                if (uvIndex !== null && uvIndex >= 0 && uvIndex < uvs.length / 2) {
                    uvsArray.push(
                        uvs[uvIndex * 2] || 0,
                        uvs[uvIndex * 2 + 1] || 0
                    );
                } else {
                    uvsArray.push(0, 0);
                }

                // Colors
                if (hasColors && vertexIndex >= 0 && vertexIndex < vertexColors.length / 3) {
                    colorsArray.push(
                        vertexColors[vertexIndex * 3] || 0.8,
                        vertexColors[vertexIndex * 3 + 1] || 0.8,
                        vertexColors[vertexIndex * 3 + 2] || 0.8
                    );
                } else {
                    colorsArray.push(0.8, 0.8, 0.8);
                }
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normalsArray, 3));
        geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvsArray, 2));

        // Add color attribute for vertex coloring
        if (hasColors) {
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorsArray, 3));
            console.log('🎨 Added color attribute to avatar geometry!');
        }

        // Compute normals if not provided
        if (normals.length === 0) {
            geometry.computeVertexNormals();
        }

        // Create material with vertex color support
        const material = new THREE.MeshLambertMaterial({
            color: hasColors ? 0xffffff : 0xcccccc,
            vertexColors: hasColors,
            side: THREE.DoubleSide,
            transparent: false,
            roughness: 0.8 // Fabric-like appearance
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        const group = new THREE.Group();
        group.add(mesh);

        if (hasColors) {
            console.log('✅ VERTEX-COLORED avatar geometry built successfully!');
        } else {
            console.log('📝 Standard avatar geometry built (no colors found)');
        }

        return group;
    }
}

// INITIALIZE AVATAR VERTEX COLOR RENDERER
console.log('🎨 Avatar Vertex Color Renderer class defined');

function initializeAvatarVertexColorRenderer() {
    if (typeof THREE !== 'undefined' && THREE.OBJLoader) {
        window.avatarVertexColorRenderer = new AvatarVertexColoringRenderer();
        console.log('✅ Avatar Vertex Color Renderer instance created globally');
        return true;
    } else {
        console.log('⏳ Waiting for THREE.js for avatar vertex color renderer...');
        return false;
    }
}

// Auto-initialize
if (!initializeAvatarVertexColorRenderer()) {
    const initInterval = setInterval(() => {
        if (initializeAvatarVertexColorRenderer()) {
            clearInterval(initInterval);
        }
    }, 100);

    setTimeout(() => {
        clearInterval(initInterval);
        if (!window.avatarVertexColorRenderer) {
            console.error('❌ Failed to initialize Avatar Vertex Color Renderer after 10 seconds');
        }
    }, 10000);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = AvatarVertexColoringRenderer;
}