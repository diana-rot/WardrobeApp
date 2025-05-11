// clothing-processor.js - Processes clothing items for RPM avatars

class ClothingProcessor {
    constructor(options = {}) {
        console.log('Initializing ClothingProcessor', options);
        this.avatarUrl = options.avatarUrl || null;
        this.apiUrl = options.apiUrl || '/api/wardrobe/process-clothing';
        this.clothingRenderer = null;
        this.clothingMapper = new ClothingMapper();
        this.activeClothingItems = new Map(); // Tracks currently applied clothing
    }

    // Set the clothing renderer instance
    setClothingRenderer(renderer) {
        if (!renderer) {
            console.error('Invalid clothing renderer provided');
            return false;
        }
        this.clothingRenderer = renderer;
        console.log('Clothing renderer set successfully');
        return true;
    }

    // Set avatar URL
    setAvatarUrl(url) {
        this.avatarUrl = url;
    }

    // Process clothing item
    async processClothingItem(itemId, imageUrl, itemType) {
        try {
            console.log('Processing clothing item:', { itemId, imageUrl, itemType });

            // Show loading indicator
            this.showProcessingMessage(`Processing ${itemType}...`);

            // Call the backend API to process the clothing
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    itemId: itemId,
                    imageUrl: imageUrl,
                    itemType: itemType
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Failed to process ${itemType}`);
            }

            const data = await response.json();
            console.log('Received clothing data:', data);

            if (!data.success) {
                throw new Error(data.error || `Failed to process ${itemType}`);
            }

            // Try to apply the clothing to the avatar
            await this.applyClothingItem({
                _id: itemId,
                label: itemType,
                file_path: imageUrl
            });

            this.showProcessingMessage(`${itemType} applied successfully!`, 'success');
            return data.modelData;
        } catch (error) {
            console.error('Error processing clothing:', error);
            this.showProcessingMessage(`Failed to process ${itemType}: ${error.message}`, 'error');
            throw error;
        }
    }

    // Apply clothing item to avatar
    async applyClothingItem(clothingItem) {
        try {
            console.log('Applying clothing item to avatar:', clothingItem);

            if (!this.clothingRenderer || !window.rpmManager) {
                console.error('Clothing renderer or RPM manager not initialized');
                this.showProcessingMessage('Avatar system not ready. Please reload the page.', 'error');
                return false;
            }

            // Get the category for this item
            const category = this.clothingMapper.getCategory(clothingItem);
            console.log(`Item category: ${category}`);

            // Remove any existing item in the same category
            if (this.activeClothingItems.has(category)) {
                const existingItemId = this.activeClothingItems.get(category);
                console.log(`Removing existing item in category ${category}: ${existingItemId}`);
                await window.rpmManager.removeClothing(existingItemId);
            }

            // Apply the item directly using RPM manager
            console.log('Loading clothing onto avatar...');
            const success = await window.rpmManager.loadClothing(
                clothingItem._id,
                clothingItem.file_path,
                clothingItem.label
            );

            if (success) {
                // Track the applied item
                this.activeClothingItems.set(category, clothingItem._id);
                console.log(`Successfully applied ${clothingItem.label} to avatar`);
                return true;
            } else {
                console.error(`Failed to apply ${clothingItem.label} to avatar`);
                this.showProcessingMessage(`Failed to apply ${clothingItem.label}`, 'error');
                return false;
            }
        } catch (error) {
            console.error('Error applying clothing item:', error);
            this.showProcessingMessage('Error applying clothing item', 'error');
            return false;
        }
    }

    // Remove clothing by item ID
    removeClothingItem(itemId) {
        try {
            if (!window.rpmManager) {
                console.error('RPM manager not initialized');
                return false;
            }

            console.log(`Removing clothing item: ${itemId}`);
            const success = window.rpmManager.removeClothing(itemId);

            if (success) {
                // Remove from tracking
                for (const [category, id] of this.activeClothingItems.entries()) {
                    if (id === itemId) {
                        this.activeClothingItems.delete(category);
                        break;
                    }
                }

                return true;
            }

            return false;
        } catch (error) {
            console.error('Error removing clothing item:', error);
            return false;
        }
    }

    // Remove all clothing
    clearAllClothing() {
        try {
            if (!window.rpmManager) {
                console.error('RPM manager not initialized');
                return false;
            }

            window.rpmManager.clearAllClothing();
            this.activeClothingItems.clear();
            return true;
        } catch (error) {
            console.error('Error clearing all clothing:', error);
            return false;
        }
    }

    // Show processing message
    showProcessingMessage(message, type = 'info') {
        // Create or update a message element
        let messageElement = document.getElementById('processing-message');

        if (!messageElement) {
            messageElement = document.createElement('div');
            messageElement.id = 'processing-message';
            messageElement.style.position = 'fixed';
            messageElement.style.bottom = '20px';
            messageElement.style.right = '20px';
            messageElement.style.padding = '10px 20px';
            messageElement.style.borderRadius = '5px';
            messageElement.style.color = 'white';
            messageElement.style.fontWeight = 'bold';
            messageElement.style.zIndex = '9999';
            document.body.appendChild(messageElement);
        }

        // Set message type styling
        switch (type) {
            case 'error':
                messageElement.style.backgroundColor = '#e74c3c';
                break;
            case 'success':
                messageElement.style.backgroundColor = '#2ecc71';
                break;
            default:
                messageElement.style.backgroundColor = '#3498db';
        }

        messageElement.textContent = message;

        // Auto-hide after 3 seconds unless it's an error
        if (type !== 'error') {
            setTimeout(() => {
                if (messageElement && messageElement.parentNode) {
                    messageElement.parentNode.removeChild(messageElement);
                }
            }, 3000);
        }
    }
}

// Initialize when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing ClothingProcessor globally');
    window.clothingProcessor = new ClothingProcessor();
});