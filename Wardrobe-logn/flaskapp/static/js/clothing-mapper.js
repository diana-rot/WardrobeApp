// ClothingMapper.js - Maps clothing types to 3D models
class ClothingMapper {
    constructor() {
        // Map clothing labels to 3D model files
        this.modelMap = {
            'T-shirt/top': '/static/models/clothing/top.glb',
            'Shirt': '/static/models/clothing/shirt.glb',
            'Pullover': '/static/models/clothing/pullover.glb',
            'Trouser': '/static/models/clothing/pants.glb',
            'Dress': '/static/models/clothing/dress.glb',
            'Coat': '/static/models/clothing/coat.glb',
            'Sandal': '/static/models/clothing/sandal.glb',
            'Sneaker': '/static/models/clothing/sneaker.glb',
            'Ankle boot': '/static/models/clothing/ankle_boot.glb',
            'Bag': '/static/models/clothing/bag.glb'
        };

        // Define fallback models for each category
        this.fallbackModels = {
            'tops': '/static/models/clothing/top.glb',
            'bottoms': '/static/models/clothing/pants.glb',
            'dresses': '/static/models/clothing/dress.glb',
            'outerwear': '/static/models/clothing/coat.glb',
            'shoes': '/static/models/clothing/shoes.glb',
            'accessories': '/static/models/clothing/bag.glb'
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
    }

    // Get 3D model path for a clothing item
    getModelPath(clothingItem) {
        const label = clothingItem.label || '';

        // First try exact match
        if (this.modelMap[label]) {
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
window.ClothingMapper = ClothingMapper;