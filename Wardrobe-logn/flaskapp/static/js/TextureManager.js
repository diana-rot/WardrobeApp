/**
 * TextureManager - Handles efficient loading and management of textures
 */
class TextureManager {
  constructor() {
    // Initialize texture loader
    this.textureLoader = new THREE.TextureLoader();
    this.textureLoader.setCrossOrigin('anonymous');

    // Setup caching
    this.textureCache = new Map();
    this.pendingLoads = new Map();

    // Track memory usage
    this.totalMemoryUsage = 0;
    this.maxMemoryUsage = 256 * 1024 * 1024; // 256MB limit

    console.log("TextureManager initialized");
  }

  /**
   * Load a texture with caching
   * @param {string} url - Texture URL
   * @param {Object} options - Loading options
   * @returns {Promise<THREE.Texture>} - Loaded texture
   */
  loadTexture(url, options = {}) {
    // Default options
    const opts = {
      anisotropy: true,
      generateMipmaps: true,
      ...options
    };

    // Check cache first
    if (this.textureCache.has(url)) {
      // Mark as recently used
      const texture = this.textureCache.get(url);
      texture.userData.lastUsed = Date.now();
      return Promise.resolve(texture);
    }

    // Check if already loading
    if (this.pendingLoads.has(url)) {
      return this.pendingLoads.get(url);
    }

    // Load new texture
    const loadPromise = new Promise((resolve, reject) => {
      this.textureLoader.load(
        url,
        (texture) => {
          // Configure texture
          if (opts.anisotropy) {
            const renderer = this.getRenderer();
            if (renderer) {
              texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
            }
          }
          texture.generateMipmaps = opts.generateMipmaps;

          // Add metadata
          texture.userData = {
            url: url,
            lastUsed: Date.now(),
            memoryUsage: this.estimateTextureMemory(texture)
          };

          // Add to cache
          this.textureCache.set(url, texture);
          this.pendingLoads.delete(url);

          // Update memory usage
          this.totalMemoryUsage += texture.userData.memoryUsage;

          // Manage cache if needed
          if (this.totalMemoryUsage > this.maxMemoryUsage) {
            this.manageCache();
          }

          resolve(texture);
        },
        undefined,
        (error) => {
          console.error(`Error loading texture: ${url}`, error);
          this.pendingLoads.delete(url);
          reject(error);
        }
      );
    });

    // Store pending load
    this.pendingLoads.set(url, loadPromise);

    return loadPromise;
  }

  /**
   * Create material from textures
   * @param {Object} textureData - Texture data with paths
   * @param {Object} materialProps - Material properties
   * @returns {Promise<THREE.Material>} - Created material
   */
  async createMaterial(textureData, materialProps) {
    try {
      // Normalize texture paths
      const texturePath = textureData.texture_path || textureData.texturePath;
      const normalMapPath = textureData.normal_map_path || textureData.normalMapPath;
      const roughnessMapPath = textureData.roughness_map_path || textureData.roughnessMapPath;

      if (!texturePath) {
        throw new Error("No texture path provided");
      }

      console.log("Texture data received:", textureData);
      console.log("Loading textures with paths:", {
        texture: texturePath,
        normalMap: normalMapPath,
        roughnessMap: roughnessMapPath
      });

      // Load all textures in parallel
      const textures = await Promise.all([
        // Load diffuse texture
        this.loadTexture(texturePath, {
          anisotropy: true,
          generateMipmaps: true
        }).catch(error => {
          console.error("Error loading diffuse texture:", error);
          throw error;
        }),

        // Load normal map if available
        normalMapPath ?
          this.loadTexture(normalMapPath, {
            anisotropy: false,
            generateMipmaps: true
          }).catch(error => {
            console.error("Error loading normal map:", error);
            return null; // Don't fail if normal map fails
          }) : Promise.resolve(null),

        // Load roughness map if available
        roughnessMapPath ?
          this.loadTexture(roughnessMapPath, {
            anisotropy: false,
            generateMipmaps: true
          }).catch(error => {
            console.error("Error loading roughness map:", error);
            return null; // Don't fail if roughness map fails
          }) : Promise.resolve(null)
      ]);

      const [diffuseMap, normalMap, roughnessMap] = textures;

      if (!diffuseMap) {
        throw new Error("Failed to load diffuse texture");
      }

      // Configure texture settings
      if (diffuseMap) {
        diffuseMap.wrapS = THREE.RepeatWrapping;
        diffuseMap.wrapT = THREE.RepeatWrapping;
      }

      if (normalMap) {
        normalMap.wrapS = THREE.RepeatWrapping;
        normalMap.wrapT = THREE.RepeatWrapping;
      }

      if (roughnessMap) {
        roughnessMap.wrapS = THREE.RepeatWrapping;
        roughnessMap.wrapT = THREE.RepeatWrapping;
      }

      // Set material properties based on material type
      const materialType = materialProps?.estimated_material || 'medium';
      const settings = this.getMaterialSettings(materialType, materialProps);

      // Create material
      const material = new THREE.MeshStandardMaterial({
        map: diffuseMap,
        normalMap: normalMap,
        roughnessMap: roughnessMap,
        roughness: settings.roughness,
        metalness: settings.metalness,
        side: THREE.DoubleSide
      });

      // Set normal scale if available
      if (normalMap) {
        material.normalScale.set(settings.normalStrength, settings.normalStrength);
      }

      // Extract primary color if available
      if (materialProps && materialProps.primary_color_rgb) {
        const color = new THREE.Color(
          materialProps.primary_color_rgb[0]/255,
          materialProps.primary_color_rgb[1]/255,
          materialProps.primary_color_rgb[2]/255
        );
        material.color = color;
      }

      console.log("Material created successfully with settings:", settings);
      return material;

    } catch (error) {
      console.error('Error creating material:', error);
      throw error; // Re-throw the error to be handled by the caller
    }
  }

  /**
   * Apply material to model
   * @param {THREE.Object3D} model - Model to apply material to
   * @param {THREE.Material} material - Material to apply
   */
  applyMaterial(model, material) {
    if (!model || !material) return;

    model.traverse((node) => {
      if (node.isMesh) {
        // Store original material
        if (!node.userData.originalMaterial) {
          node.userData.originalMaterial = node.material;
        }

        // Apply new material
        node.material = material;
      }
    });
  }

  /**
   * Restore original materials to a model
   * @param {THREE.Object3D} model - Model to restore
   */
  restoreOriginalMaterials(model) {
    if (!model) return;

    model.traverse((node) => {
      if (node.isMesh && node.userData.originalMaterial) {
        node.material = node.userData.originalMaterial;
      }
    });
  }

  /**
   * Update material settings on a model
   * @param {THREE.Object3D} model - The model to update
   * @param {Object} settings - Material settings
   */
  updateMaterialSettings(model, settings) {
    if (!model) return;

    // Find material(s) on the model
    model.traverse((node) => {
      if (node.isMesh && node.material) {
        // Update basic properties
        node.material.roughness = settings.roughness;
        node.material.metalness = settings.metalness;

        // Update normal map settings
        if (node.material.normalMap) {
          if (settings.useNormalMap) {
            node.material.normalScale.set(settings.normalStrength, settings.normalStrength);
          } else {
            node.material.normalScale.set(0, 0);
          }
        }

        // Update texture settings
        if (node.material.map) {
          // Apply pattern scale and rotation
          node.material.map.repeat.set(settings.patternScale, settings.patternScale);
          node.material.map.rotation = settings.patternRotation;
          node.material.map.needsUpdate = true;

          // Update color intensity if needed
          if (settings.useColor && node.material.color) {
            // Store original color if not already stored
            if (!node.userData.originalColor) {
              node.userData.originalColor = node.material.color.clone();
            }

            // Apply color intensity
            const color = node.userData.originalColor.clone();
            if (settings.colorIntensity !== 1.0) {
              // Convert to HSL to adjust intensity
              const hsl = {};
              color.getHSL(hsl);
              hsl.l = Math.min(1, hsl.l * settings.colorIntensity);
              color.setHSL(hsl.h, hsl.s, hsl.l);
            }
            node.material.color.copy(color);
          }

          // Sync normal map and roughness map with diffuse texture
          if (node.material.normalMap) {
            node.material.normalMap.repeat.copy(node.material.map.repeat);
            node.material.normalMap.rotation = node.material.map.rotation;
            node.material.normalMap.needsUpdate = true;
          }

          if (node.material.roughnessMap) {
            node.material.roughnessMap.repeat.copy(node.material.map.repeat);
            node.material.roughnessMap.rotation = node.material.map.rotation;
            node.material.roughnessMap.needsUpdate = true;
          }
        }

        // Mark material as needing update
        node.material.needsUpdate = true;
      }
    });
  }

  /**
   * Estimate texture memory usage
   * @param {THREE.Texture} texture - Texture to estimate
   * @returns {number} - Estimated memory usage in bytes
   */
  estimateTextureMemory(texture) {
    if (!texture || !texture.image) return 0;

    const { width, height } = texture.image;
    const bytesPerPixel = 4; // RGBA

    // Calculate base memory
    let memory = width * height * bytesPerPixel;

    // Account for mipmaps
    if (texture.generateMipmaps) {
      memory *= 1.33; // ~33% overhead for mipmaps
    }

    return memory;
  }

  /**
   * Get material settings based on material type
   */
  getMaterialSettings(materialType, materialProps) {
    // Define settings for different material types
    const settings = {
      smooth: {
        roughness: 0.3,
        metalness: 0.1,
        normalStrength: 0.5
      },
      textured: {
        roughness: 0.7,
        metalness: 0.0,
        normalStrength: 1.0
      },
      rough_textured: {
        roughness: 0.9,
        metalness: 0.0,
        normalStrength: 1.5
      },
      medium: {
        roughness: 0.5,
        metalness: 0.0,
        normalStrength: 0.8
      }
    };

    // Get settings for material type or use medium as default
    return settings[materialType] || settings.medium;
  }

  /**
   * Get renderer instance
   */
  getRenderer() {
    return window.renderer || null;
  }

  /**
   * Manage texture cache
   */
  manageCache() {
    // Create a list of all cached textures
    const textures = Array.from(this.textureCache.entries())
      .map(([url, texture]) => ({
        url,
        texture,
        lastUsed: texture.userData.lastUsed || 0,
        memoryUsage: texture.userData.memoryUsage || 0
      }));

    // Sort by last used (oldest first)
    textures.sort((a, b) => a.lastUsed - b.lastUsed);

    // Remove textures until we're under the memory limit
    let currentMemory = this.totalMemoryUsage;
    const targetMemory = this.maxMemoryUsage * 0.8; // Target 80% of max

    for (const entry of textures) {
      // Stop if we're under the target
      if (currentMemory <= targetMemory) break;

      // Remove texture
      this.textureCache.delete(entry.url);
      entry.texture.dispose();

      // Update memory usage
      currentMemory -= entry.memoryUsage;
    }

    // Update total memory usage
    this.totalMemoryUsage = currentMemory;
  }

  /**
   * Create fallback material
   */
  createFallbackMaterial(materialProps) {
    // Extract primary color if available
    const primaryColor = materialProps?.primary_color_rgb || [200, 200, 200];

    // Create color
    const color = new THREE.Color(
      primaryColor[0] / 255,
      primaryColor[1] / 255,
      primaryColor[2] / 255
    );

    // Create material
    return new THREE.MeshStandardMaterial({
      color: color,
      roughness: 0.7,
      metalness: 0.0,
      side: THREE.DoubleSide
    });
  }

  /**
   * Clear texture cache
   */
  clearCache() {
    // Dispose all textures
    this.textureCache.forEach(texture => {
      texture.dispose();
    });

    // Clear maps
    this.textureCache.clear();
    this.pendingLoads.clear();
    this.totalMemoryUsage = 0;
  }
}