# @app.route('/predict', methods=['POST'])
# @login_required
# def upload():
#     if request.method == 'POST':
#         try:
#             f = request.files['file']
#             if not f:
#                 return "No file uploaded", 400
#
#             user_id = session['user']['_id']
#             upload_dir = os.path.join('flaskapp', 'static', 'image_users', user_id)
#             os.makedirs(upload_dir, exist_ok=True)
#             file_path = os.path.join(upload_dir, secure_filename(f.filename))
#             f.save(file_path)
#
#             # Make predictions with validation
#             try:
#                 preds = model_predict(file_path, model)
#                 if not isinstance(preds, np.ndarray) or preds.size == 0:
#                     raise ValueError("Invalid prediction output")
#
#                 color_result = predict_color(file_path)
#                 if not color_result or len(color_result) < 2:
#                     raise ValueError("Invalid color prediction")
#
#                 predicted_label = np.argmax(preds)
#                 class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
#                 if predicted_label >= len(class_names):
#                     raise ValueError("Invalid predicted label index")
#
#                 result = class_names[predicted_label]
#
#                 # Save to database
#                 db.wardrobe.insert_one({
#                     'label': result,
#                     'color': ' '.join(map(str, color_result[1])),
#                     'nota': 4,
#                     'userId': user_id,
#                     'file_path': f'/static/image_users/{user_id}/{secure_filename(f.filename)}'
#                 })
#
#                 return result
#
#             except Exception as e:
#                 print(f"Prediction error: {str(e)}")
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                 return str(e), 500
#
#         except Exception as e:
#             return str(e), 500
#
#     return None



# @app.route('/outfit/day', methods=['GET', 'POST'])
# @login_required
# def get_outfit():
#     print("Debug: Entering get_outfit route")
#
#     try:
#         userId = session['user']['_id']
#         print(f"Debug: User ID: {userId}")
#         filter = {'userId': userId, 'isFavorite': 'yes'}
#         users_clothes = db.outfits.find(filter)
#
#         cityByDefault = 'Bucharest'
#         DEFAULT_RATING = 4
#
#         # Default to show generator and hide outfits
#         show_generator = True
#         show_outfits = False
#         success_message = None
#         error_message = None
#
#         # Define available outfit combinations
#         result_outfit = [
#             'Dress_Sandal', 'T-shirt/top_Trouser_Sneaker', 'Shirt_Trouser',
#             'Shirt_Trouser_Sneaker', 'Dress_Sandal_Coat', 'T-shirt/top_Trouser',
#             'Shirt_Trouser_Coat', 'Shirt_Trouser_Coat', 'Dress_Ankle-boot_Coat',
#             'Pullover_Trouser_Ankle-boot', 'Dress_Sneaker', 'Shirt_Trouser_Sandal',
#             'Dress_Sandal_Bag'
#         ]
#
#         # Initialize city if not exists
#         filter = {'userId': userId}
#         if db.city.count_documents(filter) == 0:
#             print(f"Debug: Creating new city entry for user {userId}")
#             db.city.insert_one({'name': cityByDefault, 'userId': userId})
#
#         # Get weather data
#         cities = db.city.find(filter)
#         weather_data = []
#         url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=aa73cad280fbd125cc7073323a135efa'
#
#         for city in cities:
#             try:
#                 print(f"Debug: Fetching weather for {city['name']}")
#                 r = requests.get(url.format(city['name']), timeout=5).json()
#                 weather = {
#                     'city': city['name'],
#                     'temperature': r['main']['temp'],
#                     'description': r['weather'][0]['description'],
#                     'icon': r['weather'][0]['icon'],
#                 }
#                 weather_data.append(weather)
#             except Exception as e:
#                 print(f"Error fetching weather for {city['name']}: {e}")
#                 weather_data.append({
#                     'city': city['name'],
#                     'temperature': 20,
#                     'description': '',
#                     'icon': ''
#                 })
#
#         # Ensure we have 3 weather options
#         while len(weather_data) < 3:
#             weather_data.append({
#                 'city': cityByDefault,
#                 'temperature': 20,
#                 'description': '',
#                 'icon': ''
#             })
#
#         city1, city2, city3 = weather_data[:3]
#         outfit1, outfit2, outfit3 = [], [], []
#
#         if request.method == 'POST':
#             print("Debug: Processing POST request")
#
#             # Handle outfit selection
#             option = request.form.get('options')
#             if option:
#                 print(f"Debug: Selected option: {option}")
#                 filter_lookup = {'userId': userId, 'outfitNo': option}
#                 outfit_doc = db.outfits.find_one(filter_lookup, sort=[('_id', -1)])
#
#                 if outfit_doc:
#                     # Update outfit pieces ratings
#                     for piece in outfit_doc['outfit']:
#                         try:
#                             current_piece = db.wardrobe.find_one({'_id': piece['_id']})
#                             current_rating = current_piece.get('nota',
#                                                                DEFAULT_RATING) if current_piece else DEFAULT_RATING
#
#                             db.wardrobe.update_one(
#                                 {'_id': piece['_id']},
#                                 {'$set': {'nota': current_rating + 1}},
#                                 upsert=True
#                             )
#                         except Exception as e:
#                             print(f"Error updating piece rating: {str(e)}")
#
#                     try:
#                         # Update outfit rating
#                         current_outfit_rating = outfit_doc.get('nota', DEFAULT_RATING)
#                         db.outfits.update_one(
#                             {'_id': outfit_doc['_id']},
#                             {
#                                 '$set': {
#                                     'nota': current_outfit_rating + 1,
#                                     'isFavorite': 'yes'
#                                 }
#                             }
#                         )
#                         # Show success message and hide outfits
#                         success_message = "Outfit has been saved to your favorites!"
#                         show_outfits = False
#                         return render_template(
#                             'outfit_of_the_day.html',
#                             success_message=success_message,
#                             show_generator=show_generator,
#                             show_outfits=show_outfits,
#                             city1=city1,
#                             city2=city2,
#                             city3=city3
#                         )
#                     except Exception as e:
#                         print(f"Error updating outfit rating: {str(e)}")
#                         error_message = "Error saving outfit. Please try again."
#
#             # Generate new outfits
#             include_weather = request.form.get('weather') == 'yes'
#             city = request.form.get('city')
#             event = request.form.get('events')
#             temperature = 20  # Default temperature
#
#             print(f"Debug: Form data - weather: {include_weather}, city: {city}, event: {event}")
#
#             if include_weather and city:
#                 selected_weather = next(
#                     (w for w in weather_data if w['city'] == city),
#                     {'temperature': 20}
#                 )
#                 temperature = selected_weather['temperature']
#
#             try:
#                 loaded_classifier = joblib.load("./random_forest.joblib")
#                 features = prepare_features(include_weather, event, temperature)
#                 result_forest = loaded_classifier.predict([features])
#                 index_of_outfit = result_forest[0]
#                 outfit_combination = result_outfit[index_of_outfit]
#                 filters_outfits = outfit_combination.split('_')
#
#                 print(f"Debug: Generated outfit combination: {outfit_combination}")
#
#                 # Generate three outfits
#                 for i, outfit_list in enumerate([outfit1, outfit2, outfit3]):
#                     outfit_pieces = []
#                     for filter_name in filters_outfits:
#                         clothes = list(db.wardrobe.find({
#                             'userId': userId,
#                             'label': filter_name
#                         }).sort('nota', -1))
#
#                         if clothes:
#                             index = min(i, len(clothes) - 1)
#                             piece = clothes[index]
#                             if not piece.get('file_path'):
#                                 piece['file_path'] = None
#                             if 'nota' not in piece:
#                                 piece['nota'] = DEFAULT_RATING
#                                 db.wardrobe.update_one(
#                                     {'_id': piece['_id']},
#                                     {'$set': {'nota': DEFAULT_RATING}}
#                                 )
#                             outfit_pieces.append(piece)
#
#                     if outfit_pieces:
#                         outfit_doc = {
#                             'outfit': outfit_pieces,
#                             'userId': userId,
#                             'nota': DEFAULT_RATING,
#                             'outfitNo': f'piece{i + 1}',
#                             'isFavorite': 'no',
#                             'created_at': datetime.now()
#                         }
#                         db.outfits.insert_one(outfit_doc)
#
#                         if i == 0:
#                             outfit1 = outfit_pieces
#                         elif i == 1:
#                             outfit2 = outfit_pieces
#                         else:
#                             outfit3 = outfit_pieces
#
#                 show_outfits = True
#
#             except Exception as e:
#                 print(f"Error generating outfits: {e}")
#                 error_message = "Error generating outfits. Please try again."
#
#         print("Debug: Rendering template")
#         return render_template(
#             'outfit_of_the_day.html',
#             outfit1=outfit1,
#             outfit2=outfit2,
#             outfit3=outfit3,
#             city1=city1,
#             city2=city2,
#             city3=city3,
#             show_generator=show_generator,
#             show_outfits=show_outfits,
#             success_message=success_message,
#             error_message=error_message
#         )
#
#     except Exception as e:
#         print(f"Error in get_outfit: {str(e)}")
#         return render_template(
#             'outfit_of_the_day.html',
#             error_message="An error occurred. Please try again.",
#             show_generator=True,
#             show_outfits=False,
#             city1={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
#             city2={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''},
#             city3={'city': cityByDefault, 'temperature': 20, 'description': '', 'icon': ''}
#         )

# # Enhanced Avatar generation imports
# import mediapipe as mp
# import cv2
# import numpy as np
# from PIL import Image
# import json
# import base64
# from io import BytesIO
# import colorsys
# from sklearn.cluster import KMeans
#
# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# def extract_facial_features(image_path):
#     """Extract facial features using MediaPipe Face Mesh."""
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image_rgb)
#
#     if not results.multi_face_landmarks:
#         raise ValueError("No face detected in the image")
#
#     landmarks = results.multi_face_landmarks[0]
#
#     # Extract key facial features
#     features = {
#         'face_width': landmarks.landmark[234].x - landmarks.landmark[454].x,
#         'face_height': landmarks.landmark[152].y - landmarks.landmark[10].y,
#         'eye_distance': landmarks.landmark[33].x - landmarks.landmark[263].x,
#         'nose_length': landmarks.landmark[6].y - landmarks.landmark[94].y,
#         'mouth_width': landmarks.landmark[61].x - landmarks.landmark[291].x
#     }
#
#     return features
#
# def analyze_skin_tone(image_path):
#     """Analyze skin tone from the uploaded image."""
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image_rgb)
#
#     if not results.multi_face_landmarks:
#         raise ValueError("No face detected in the image")
#
#     # Get face region
#     landmarks = results.multi_face_landmarks[0]
#     face_points = np.array([[int(l.x * image.shape[1]), int(l.y * image.shape[0])]
#                           for l in landmarks.landmark])
#
#     # Create mask for face region
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     cv2.fillConvexPoly(mask, face_points, 255)
#
#     # Get average skin color
#     face_region = cv2.bitwise_and(image, image, mask=mask)
#     skin_color = cv2.mean(face_region, mask=mask)[:3]
#
#     return [int(c) for c in skin_color]
#
# @app.route('/api/avatar/generate', methods=['POST'])
# @login_required
# def generate_avatar():
#     try:
#         if 'photo' not in request.files:
#             return jsonify({'error': 'No photo uploaded'}), 400
#
#         photo = request.files['photo']
#         gender = request.form.get('gender', 'female')
#
#         if photo.filename == '':
#             return jsonify({'error': 'No selected file'}), 400
#
#         # Save uploaded photo
#         filename = secure_filename(photo.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         photo.save(filepath)
#
#         # Extract facial features
#         features = extract_facial_features(filepath)
#
#         # Analyze skin tone
#         skin_color = analyze_skin_tone(filepath)
#
#         # Prepare avatar data
#         avatar_data = {
#             'model_path': MODEL_PATHS[gender]['model'],
#             'textures': MODEL_PATHS[gender]['textures'],
#             'features': features,
#             'skin_color': skin_color,
#             'gender': gender
#         }
#
#         # Save avatar data to user's profile
#         user_id = session.get('user_id')
#         if user_id:
#             db.users.update_one(
#                 {'_id': ObjectId(user_id)},
#                 {'$set': {'avatar_data': avatar_data}}
#             )
#
#         return jsonify(avatar_data)
#
#     except Exception as e:
#         app.logger.error(f"Error in generate_avatar: {str(e)}")
#         return jsonify({'error': str(e)}), 500
#
# @app.route('/api/avatar/get', methods=['GET'])
# @login_required
# def get_user_avatar_data():
#     """Get user's current avatar data"""
#     try:
#         user_id = session['user']['_id']
#         avatar_doc = db.avatars.find_one({'userId': user_id})
#
#         if not avatar_doc:
#             return jsonify({'error': 'No avatar found'}), 404
#
#         # Get the avatar data
#         avatar_data = avatar_doc.get('avatarData', {})
#
#         # Add the model path if not present
#         if 'model_path' not in avatar_data and 'gender' in avatar_data:
#             avatar_data['model_path'] = MODEL_PATHS[avatar_data['gender'].lower()]
#
#         return jsonify({
#             'success': True,
#             'avatarData': avatar_data
#         })
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
# # New endpoint for trying on clothes
# @app.route('/api/avatar/try-on', methods=['POST'])
# @login_required
# def try_on_clothes():
#     """Try on clothes from wardrobe on the avatar"""
#     try:
#         user_id = session['user']['_id']
#         item_id = request.json.get('itemId')
#
#         if not item_id:
#             return jsonify({'error': 'No item ID provided'}), 400
#
#         # Get the clothing item from the wardrobe
#         item = db.wardrobe.find_one({'_id': ObjectId(item_id), 'userId': user_id})
#
#         if not item:
#             return jsonify({'error': 'Item not found'}), 404
#
#         # Get the avatar data
#         avatar_data = db.avatars.find_one({'userId': user_id})
#
#         if not avatar_data:
#             return jsonify({'error': 'No avatar found. Please create an avatar first.'}), 404
#
#         # Return the item data for the avatar to wear
#         return jsonify({
#             'success': True,
#             'item': {
#                 'id': str(item['_id']),
#                 'type': item.get('label', '').lower(),
#                 'color': item.get('color', ''),
#                 'image_url': normalize_path(item.get('file_path', ''))
#             }
#         })
#
#     except Exception as e:
#         print(f"Error in try_on_clothes: {str(e)}")
#         return jsonify({'error': str(e)}), 500
#
# @app.route('/model-inspector')
# @login_required
# def model_inspector():
#     return render_template('model_inspector.html')
#
#
# # Update avatar data
# @app.route('/api/avatar/update', methods=['POST'])
# @login_required
# def update_avatar():
#     try:
#         user_id = session['user']['_id']
#         avatar_data = request.json
#
#         if not avatar_data:
#             return jsonify({'error': 'No avatar data provided'}), 400
#
#         # Update avatar document
#         result = db.avatars.update_one(
#             {'userId': user_id},
#             {
#                 '$set': {
#                     'avatarData': avatar_data,
#                     'updatedAt': datetime.now()
#                 }
#             },
#             upsert=True
#         )
#
#         return jsonify({
#             'success': True,
#             'message': 'Avatar updated successfully'
#         })
#
#     except Exception as e:
#         print(f"Error updating avatar: {str(e)}")
#         return jsonify({'error': str(e)}), 500
#
#
