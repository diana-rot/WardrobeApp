// static/js/clothing-processor.js

// Class to process 2D clothing images into 3D models for RPM avatars
class ClothingProcessor {
    constructor(options = {}) {
        this.avatarUrl = options.avatarUrl || null;
        this.apiUrl = options.apiUrl || '/api/wardrobe/process-clothing';
    }

    // Process a single clothing item
    async processClothingItem(itemId, imageUrl, itemType) {
        try {
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

            if (!data.success) {
                throw new Error(data.error || `Failed to process ${itemType}`);
            }

            // Return the processed model data
            return data.modelData;
        } catch (error) {
            console.error('Error processing clothing:', error);
            this.showProcessingMessage(`Failed to process ${itemType}`, 'error');
            throw error;
        }
    }

    // Apply clothing to avatar
    async applyClothingToAvatar(clothingData) {
        // Implementation will depend on how you integrate with the RPM API
        // For now, we'll just return the data
        return clothingData;
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
    window.clothingProcessor = new ClothingProcessor();
});