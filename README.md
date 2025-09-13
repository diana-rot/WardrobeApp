# Wardrobe  Application

The Wardrobe Application is a Flask-based web app designed to help users organize and manage their clothing items, predict suitable outfits based on weather and events, and maintain a personal wardrobe collection.

## Features

1. **User Authentication**:
   - Login and registration system for personalized wardrobe management.
2. **Wardrobe Management**:
   - Add, view, and update clothing items in the wardrobe.
   - Classify clothing items using an AI model.
3. **Outfit Suggestions**:
   - Suggest daily outfits based on weather, temperature, and user preferences.
4. **Weather Integration**:
   - Fetch real-time weather data for selected cities.
   - Use weather conditions to influence outfit suggestions.
5. **Outfit Ratings and Favorites**:
   - Rate outfits and mark favorites for quick access.

## Project Structure

```
project/
├── app/
│   ├── __init__.py        # Flask app factory
│   ├── routes.py          # Flask routes
│   ├── services.py        # MongoDB interaction logic
│   ├── utils.py           # Utility functions
├── models/
│   ├── model.py           # AI model for clothing classification
├── static/                # Static files (CSS, JS, images)
├── templates/             # HTML templates
├── uploads/               # Uploaded images
├── run.py                 # Application entry point
```

## Requirements

- Python 3.8+
- MongoDB

### Python Libraries
Install dependencies using:
```bash
pip install -r requirements.txt
```

Dependencies include:
- `Flask`
- `pymongo`
- `requests`
- `tensorflow`
- `scikit-learn`
- `opencv-python`

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/wardrobe-management-app.git
   cd wardrobe-management-app
   ```

2. **Set Up MongoDB**:
   - Ensure MongoDB is running on `localhost:27017`.
   - Create a database named `user_login_system_test`.

3. **Run the Application**:
   ```bash
   python run.py
   ```
   Access the app at `http://127.0.0.1:5000/`.

4. **Configure Weather API**:
   - Use OpenWeatherMap API for weather data.
   - Add your API key in `app/routes.py` under the `api_key` variable.

## Key Functionalities

### 1. Wardrobe Management
- **Add Items**:
  Upload images of clothing items to the wardrobe.
- **Classify Items**:
  Use an AI model to classify clothing into categories like "T-shirt", "Dress", etc.

### 2. Outfit Suggestions
- **Weather-Based Suggestions**:
  Get recommended outfits based on real-time weather data.
- **Event-Based Suggestions**:
  Choose events like "work" or "travel" to customize outfit suggestions.

### 3. Weather Integration
- **City Management**:
  Add and manage cities for weather data.
- **Real-Time Weather**:
  View temperature, weather description, and icons for selected cities.

### 4. Favorite Outfits
- Mark outfits as favorites for quick reference.
- Track outfit ratings and improve suggestions over time.

## Extending the Application

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with detailed changes.


