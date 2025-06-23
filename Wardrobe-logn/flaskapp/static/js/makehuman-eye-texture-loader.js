/**
 * Eye Texture Loader for Existing MakeHuman Textures
 * This fixes eye rendering by properly loading your existing eye texture files
 */

class MakeHumanEyeTextureLoader {
    constructor(avatarManager) {
        this.avatarManager = avatarManager;
        this.textureLoader = new THREE.TextureLoader();
        this.loadedTextures = new Map();
        this.currentEyeColor = 'brown';

        // Map your eye colors to the actual texture files
        this.eyeTextureFiles = {
            'brown': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'blue': '/static/models/makehuman/bodies/male/textures/blue_eye.png',
            // Add more mappings as you have more eye textures
            'dark-brown': '/static/models/makehuman/bodies/male/textures/brown_eye.png', // fallback
            'light-brown': '/static/models/makehuman/bodies/male/textures/brown_eye.png', // fallback
            'hazel': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'amber': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'gold': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'light-blue': '/static/models/makehuman/bodies/male/textures/blue_eye.png',
            'dark-blue': '/static/models/makehuman/bodies/male/textures/blue_eye.png',
            'sky-blue': '/static/models/makehuman/bodies/male/textures/blue_eye.png',
            'green': '/static/models/makehuman/bodies/male/textures/brown_eye.png', // You might have green_eye.png
            'light-green': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'dark-green': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'emerald': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'purple': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'violet': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'gray': '/static/models/makehuman/bodies/male/textures/brown_eye.png',
            'silver': '/static/models/makehuman/bodies/male/textures/brown_eye.png'
        };

        this.init();
    }

    init() {
        console.log('Initializing MakeHuman Eye Texture Loader...');
        this.preloadEyeTextures();
    }

    /**
     * Preload all available eye textures
     */
    async preloadEyeTextures() {
        console.log('Preloading MakeHuman eye textures...');

        for (const [colorKey, texturePath] of Object.entries(this.eyeTextureFiles)) {
            try {
                await this.loadEyeTexture(colorKey, texturePath);
                console.log(`Loaded eye texture: ${colorKey}`);
            } catch (error) {
                console.warn(`Failed to load eye texture ${colorKey}:`, error);
                // Create fallback texture
                this.createFallbackTexture(colorKey);
            }
        }

        console.log('Eye texture preloading complete');
    }

    /**
     * Load a specific eye texture
     */
    loadEyeTexture(colorKey, texturePath) {
        return new Promise((resolve, reject) => {
            this.textureLoader.load(
                texturePath,
                (texture) => {
                    // Configure texture for GLB models
                    texture.wrapS = THREE.ClampToEdgeWrapping;
                    texture.wrapT = THREE.ClampToEdgeWrapping;
                    texture.flipY = false; // Critical for GLB models
                    texture.generateMipmaps = true;
                    texture.minFilter = THREE.LinearMipmapLinearFilter;
                    texture.magFilter = THREE.LinearFilter;

                    this.loadedTextures.set(colorKey, texture);
                    resolve(texture);
                },
                (progress) => {
                    console.log(`Loading ${colorKey} eye texture: ${(progress.loaded / progress.total * 100)}%`);
                },
                (error) => {
                    console.error(`Failed to load eye texture ${colorKey}:`, error);
                    reject(error);
                }
            );
        });
    }

    /**
     * Create fallback texture if original fails to load
     */
    createFallbackTexture(colorKey) {
        console.log(`Creating fallback texture for ${colorKey}`);

        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');

        // Eye color mapping for fallbacks
        const eyeColorValues = {
            'brown': '#8B4513',
            'blue': '#4A90E2',
            'green': '#50C878',
            'gray': '#708090',
            'dark-brown': '#4A2C17',
            'light-brown': '#D2B48C',
            'hazel': '#8B6914',
            'amber': '#FFBF00'
        };

        const eyeColor = eyeColorValues[colorKey] || '#8B4513';

        // Draw realistic eye
        this.drawRealisticEye(ctx, canvas.width / 2, canvas.height / 2, canvas.width / 2, eyeColor);

        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        texture.flipY = false;

        this.loadedTextures.set(colorKey, texture);
    }

    /**
     * Draw a realistic eye on canvas
     */
    drawRealisticEye(ctx, centerX, centerY, radius, irisColor) {
        // Clear canvas
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        // Draw sclera (white part)
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.9, 0, Math.PI * 2);
        ctx.fill();

        // Add subtle sclera shading
        const scleraGradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, radius * 0.9
        );
        scleraGradient.addColorStop(0, '#FFFFFF');
        scleraGradient.addColorStop(0.7, '#F8F8F8');
        scleraGradient.addColorStop(1, '#E8E8E8');

        ctx.fillStyle = scleraGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.9, 0, Math.PI * 2);
        ctx.fill();

        // Draw iris
        const irisRadius = radius * 0.4;
        const irisGradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, irisRadius
        );
        irisGradient.addColorStop(0, this.lightenColor(irisColor, 30));
        irisGradient.addColorStop(0.3, irisColor);
        irisGradient.addColorStop(0.7, this.darkenColor(irisColor, 20));
        irisGradient.addColorStop(1, this.darkenColor(irisColor, 40));

        ctx.fillStyle = irisGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, irisRadius, 0, Math.PI * 2);
        ctx.fill();

        // Add iris texture lines
        ctx.strokeStyle = this.darkenColor(irisColor, 50);
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;

        for (let i = 0; i < 20; i++) {
            const angle = (i / 20) * Math.PI * 2;
            const startRadius = irisRadius * 0.3;
            const endRadius = irisRadius * 0.9;

            ctx.beginPath();
            ctx.moveTo(
                centerX + Math.cos(angle) * startRadius,
                centerY + Math.sin(angle) * startRadius
            );
            ctx.lineTo(
                centerX + Math.cos(angle) * endRadius,
                centerY + Math.sin(angle) * endRadius
            );
            ctx.stroke();
        }

        ctx.globalAlpha = 1;

        // Draw pupil
        const pupilRadius = radius * 0.15;
        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.arc(centerX, centerY, pupilRadius, 0, Math.PI * 2);
        ctx.fill();

        // Add specular highlights
        const highlight1Gradient = ctx.createRadialGradient(
            centerX - pupilRadius * 0.5, centerY - pupilRadius * 0.5, 0,
            centerX - pupilRadius * 0.5, centerY - pupilRadius * 0.5, pupilRadius * 2
        );
        highlight1Gradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
        highlight1Gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

        ctx.fillStyle = highlight1Gradient;
        ctx.beginPath();
        ctx.arc(centerX - pupilRadius * 0.5, centerY - pupilRadius * 0.5, pupilRadius * 2, 0, Math.PI * 2);
        ctx.fill();

        // Secondary highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.beginPath();
        ctx.arc(centerX + pupilRadius * 1.5, centerY + pupilRadius, pupilRadius * 0.7, 0, Math.PI * 2);
        ctx.fill();
    }

    /**
     * Apply eye texture to avatar model
     */
    applyEyeTextureToAvatar(avatarModel, eyeColor = 'brown') {
        if (!avatarModel) {
            console.warn('No avatar model provided for eye texture application');
            return false;
        }

        console.log(`Applying MakeHuman eye texture: ${eyeColor}`);

        const eyeTexture = this.loadedTextures.get(eyeColor);
        if (!eyeTexture) {
            console.warn(`Eye texture for color ${eyeColor} not found`);
            return false;
        }

        let eyesFound = 0;

        // Traverse the avatar model to find eye-related meshes
        avatarModel.traverse((child) => {
            if (child.isMesh && child.material) {
                const name = child.name.toLowerCase();
                const materialName = child.material.name ? child.material.name.toLowerCase() : '';

                // Check for eye-related mesh names (MakeHuman specific)
                const isEyeMesh = this.isMakeHumanEyeMesh(name, materialName);

                if (isEyeMesh) {
                    console.log(`Found MakeHuman eye mesh: ${child.name}, material: ${materialName}`);

                    // Create proper eye material
                    const eyeMaterial = new THREE.MeshStandardMaterial({
                        map: eyeTexture,
                        transparent: false,
                        alphaTest: 0.1,
                        side: THREE.FrontSide,

                        // Eye-specific material properties
                        metalness: 0.1,
                        roughness: 0.3,

                        // Ensure proper lighting
                        flatShading: false,

                        // Color adjustments
                        color: 0xffffff,
                        emissive: 0x000000
                    });

                    // Apply material
                    if (Array.isArray(child.material)) {
                        // Handle multi-material meshes
                        child.material = child.material.map(mat => {
                            const matName = mat.name ? mat.name.toLowerCase() : '';
                            return this.isMakeHumanEyeMesh('', matName) ? eyeMaterial : mat;
                        });
                    } else {
                        child.material = eyeMaterial;
                    }

                    // Ensure proper geometry
                    if (child.geometry) {
                        child.geometry.computeVertexNormals();
                    }

                    // Enable shadows
                    child.castShadow = true;
                    child.receiveShadow = true;

                    eyesFound++;
                }
            }
        });

        if (eyesFound > 0) {
            console.log(`Successfully applied MakeHuman eye textures to ${eyesFound} mesh(es)`);
            this.currentEyeColor = eyeColor;
            return true;
        } else {
            console.warn('No MakeHuman eye meshes found in avatar model');
            // Try generic approach
            return this.applyGenericEyeTexture(avatarModel, eyeColor);
        }
    }

    /**
     * Check if mesh is a MakeHuman eye mesh
     */
    isMakeHumanEyeMesh(meshName, materialName) {
        const eyeKeywords = [
            'eye', 'eyes', 'eyeball', 'eyeballs', 'iris', 'pupil', 'sclera',
            'cornea', 'eye_l', 'eye_r', 'eyeball_l', 'eyeball_r',
            'lefteye', 'righteye', 'leye', 'reye'
        ];

        const nameToCheck = (meshName + ' ' + materialName).toLowerCase();
        return eyeKeywords.some(keyword => nameToCheck.includes(keyword));
    }

    /**
     * Generic eye texture application
     */
    applyGenericEyeTexture(avatarModel, eyeColor) {
        console.log('Trying generic eye texture application...');

        const eyeTexture = this.loadedTextures.get(eyeColor);
        if (!eyeTexture) return false;

        let applied = 0;

        avatarModel.traverse((child) => {
            if (child.isMesh && child.material) {
                // Look for materials that might be eyes based on their properties
                const material = Array.isArray(child.material) ? child.material[0] : child.material;

                if (material && material.map) {
                    const mapSrc = material.map.image ? material.map.image.src : '';

                    // Check if the material uses an eye texture
                    if (mapSrc.includes('eye') || mapSrc.includes('brown_eye') || mapSrc.includes('blue_eye')) {
                        console.log('Found material with eye texture, updating...', child.name);

                        const newMaterial = material.clone();
                        newMaterial.map = eyeTexture;
                        newMaterial.needsUpdate = true;

                        if (Array.isArray(child.material)) {
                            child.material = child.material.map(mat =>
                                (mat.map && mat.map.image && mat.map.image.src.includes('eye')) ? newMaterial : mat
                            );
                        } else {
                            child.material = newMaterial;
                        }

                        applied++;
                    }
                }
            }
        });

        return applied > 0;
    }

    /**
     * Update eye color
     */
    updateEyeColor(eyeColor) {
        if (!this.avatarManager || !this.avatarManager.avatarModel) {
            console.warn('No avatar model available for eye color update');
            return false;
        }

        return this.applyEyeTextureToAvatar(this.avatarManager.avatarModel, eyeColor);
    }

    /**
     * Get the correct texture path based on gender
     */
    getEyeTexturePath(eyeColor, gender = 'male') {
        const basePath = `/static/models/makehuman/bodies/${gender}/textures/`;

        // Map colors to actual available files
        const fileMap = {
            'brown': 'brown_eye.png',
            'blue': 'blue_eye.png',
            // Add more mappings based on your available files
        };

        const fileName = fileMap[eyeColor] || 'brown_eye.png';
        return basePath + fileName;
    }

    /**
     * Color utility functions
     */
    lightenColor(hex, percent) {
        const num = parseInt(hex.slice(1), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }

    darkenColor(hex, percent) {
        const num = parseInt(hex.slice(1), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;
        return "#" + (0x1000000 + (R > 255 ? 255 : R < 0 ? 0 : R) * 0x10000 +
            (G > 255 ? 255 : G < 0 ? 0 : G) * 0x100 +
            (B > 255 ? 255 : B < 0 ? 0 : B)).toString(16).slice(1);
    }

    /**
     * Cleanup
     */
    dispose() {
        this.loadedTextures.forEach(texture => {
            if (texture.dispose) texture.dispose();
        });
        this.loadedTextures.clear();
        console.log('MakeHuman Eye Texture Loader disposed');
    }
}

// Integration with your existing system
function integrateMakeHumanEyeLoader() {
    let makeHumanEyeLoader = null;

    const initLoader = () => {
        if (window.avatarManager) {
            makeHumanEyeLoader = new MakeHumanEyeTextureLoader(window.avatarManager);
            window.makeHumanEyeLoader = makeHumanEyeLoader;

            console.log('MakeHuman Eye Texture Loader initialized');
            return true;
        }
        return false;
    };

    if (!initLoader()) {
        const checkInterval = setInterval(() => {
            if (initLoader()) {
                clearInterval(checkInterval);
            }
        }, 1000);

        setTimeout(() => clearInterval(checkInterval), 10000);
    }
}

// Usage in your existing code:
/*
// Initialize after avatar manager is ready
integrateMakeHumanEyeLoader();

// Apply eye texture when avatar loads
if (window.makeHumanEyeLoader && avatarModel) {
    window.makeHumanEyeLoader.applyEyeTextureToAvatar(avatarModel, 'brown');
}

// Update eye color
if (window.makeHumanEyeLoader) {
    window.makeHumanEyeLoader.updateEyeColor('blue');
}
*/

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        integrateMakeHumanEyeLoader();
    }, 3000);
});

// Export
if (typeof window !== 'undefined') {
    window.MakeHumanEyeTextureLoader = MakeHumanEyeTextureLoader;
    window.integrateMakeHumanEyeLoader = integrateMakeHumanEyeLoader;
}