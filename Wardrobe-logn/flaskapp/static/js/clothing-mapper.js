// ClothingMapper.js - Maps clothing types to 3D models
class ClothingMapper {
    constructor() {
        console.log('Initializing ClothingMapper');
        
        // Define default model path
        const defaultModel = '/static/models/clothing/top.glb';
        
        // Map clothing labels to 3D model files
        this.modelMap = {
            'T-shirt/top': defaultModel,
            'Shirt': defaultModel,
            'Pullover': defaultModel,
            'Trouser': defaultModel,
            'Dress': defaultModel,
            'Coat': defaultModel,
            'Sandal': defaultModel,
            'Sneaker': defaultModel,
            'Ankle boot': defaultModel,
            'Bag': defaultModel
        };

        // Define fallback models for each category
        this.fallbackModels = {
            'tops': defaultModel,
            'bottoms': defaultModel,
            'dresses': defaultModel,
            'outerwear': defaultModel,
            'shoes': defaultModel,
            'accessories': defaultModel
        };

        // Define clothing categories
        this.categories = {
            'tops': ['T-shirt/top', 'Shirt', 'Pullover'],
            'bottoms': ['Trouser'],
            'dresses': ['Dress'],
            'outerwear': ['Coat'],
            'shoes': ['Sandal', 'Sneaker', 'Ankle boot'],
            'accessories': ['Bag']
        };

        // Store default model for reference
        this.defaultModel = defaultModel;

        // Verify model exists
        this.verifyModelExists(defaultModel);
    }

    // Verify if a model file exists
    async verifyModelExists(modelPath) {
        try {
            const response = await fetch(modelPath, { method: 'HEAD' });
            if (!response.ok) {
                console.warn(`Model not found: ${modelPath}`);
                return false;
            }
            return true;
        } catch (error) {
            console.warn(`Error checking model: ${modelPath}`, error);
            return false;
        }
    }

    // Get 3D model path for a clothing item
    getModelPath(clothingItem) {
        console.log('Getting model path for:', clothingItem);

        if (!clothingItem) {
            console.warn('No clothing item provided');
            return this.fallbackModels['tops'];
        }

        const label = clothingItem.label || '';
        console.log(`Looking for model for label: "${label}"`);

        // First try exact match
        if (this.modelMap[label]) {
            console.log(`Found exact model match: ${this.modelMap[label]}`);
            return this.modelMap[label];
        }

        // If no exact match, get category and use fallback
        const category = this.getCategory(clothingItem);
        return this.fallbackModels[category] || this.fallbackModels['tops'];
    }

    // Get category for a clothing item
    getCategory(clothingItem) {
        const label = clothingItem.label || '';

        for (const [category, labels] of Object.entries(this.categories)) {
            if (labels.includes(label)) {
                return category;
            }
        }

        return 'unknown';
    }

    // Get position and scale adjustments for a clothing type
    getPositionAndScale(category) {
        const defaults = {
            position: { x: 0, y: 1.2, z: 0 },
            scale: { x: 1, y: 1, z: 1 },
            rotation: { x: 0, y: 0, z: 0 }
        };

        switch(category) {
            case 'tops':
                return {
                    position: { x: 0, y: 1.4, z: 0 },
                    scale: { x: 1, y: 1, z: 1 },
                    rotation: { x: 0, y: 0, z: 0 }
                };
            case 'bottoms':
                return {
                    position: { x: 0, y: 0.8, z: 0 },
                    scale: { x: 1, y: 1, z: 1 },
                    rotation: { x: 0, y: 0, z: 0 }
                };
            case 'dresses':
                return {
                    position: { x: 0, y: 1.1, z: 0 },
                    scale: { x: 1, y: 1, z: 1 },
                    rotation: { x: 0, y: 0, z: 0 }
                };
            case 'outerwear':
                return {
                    position: { x: 0, y: 1.3, z: 0.05 },
                    scale: { x: 1.1, y: 1, z: 1 },
                    rotation: { x: 0, y: 0, z: 0 }
                };
            case 'shoes':
                return {
                    position: { x: 0, y: 0.05, z: 0 },
                    scale: { x: 1, y: 1, z: 1 },
                    rotation: { x: 0, y: 0, z: 0 }
                };
            case 'accessories':
                return {
                    position: { x: 0.3, y: 1.0, z: 0.2 },
                    scale: { x: 0.8, y: 0.8, z: 0.8 },
                    rotation: { x: 0, y: 0, z: 0 }
                };
            default:
                return defaults;
        }
    }

    // Get all clothing types
    getAllClothingTypes() {
        return Object.keys(this.modelMap);
    }

    // Get all categories
    getAllCategories() {
        return Object.keys(this.categories);
    }

    // Get clothing types by category
    getTypesByCategory(category) {
        return this.categories[category] || [];
    }
}

// Make available globally
if (typeof window !== 'undefined') {
    window.ClothingMapper = ClothingMapper;
    console.log('ClothingMapper initialized globally');
} else {
    // For non-browser environments
    module.exports = ClothingMapper;
}