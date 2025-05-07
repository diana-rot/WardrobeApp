from flask import jsonify, request, session, redirect
from passlib.hash import pbkdf2_sha256
from flaskapp import db
import uuid

class User:

  def start_session(self, user):
    del user['password']
    session['logged_in'] = True
    session['user'] = user
    session['user_id'] = str(user['_id'])
    print(session['user'])
    return jsonify(user), 200

  def signup(self):
    print(request.form)

    # Create the user object
    user = {
      "_id": uuid.uuid4().hex,
      "name": request.form.get('name'),
      "email": request.form.get('email'),
      "password": request.form.get('password'),
    }

    # Encrypt the password
    user['password'] = pbkdf2_sha256.encrypt(user['password'])
    # Check for existing email address
    if db.users.find_one({ "email": user['email'] }):
      return jsonify({ "error": "Email address already in use" }), 400

    if db.users.insert_one(user):
      return self.start_session(user)

    return jsonify({ "error": "Signup failed" }), 400
  
  def signout(self):
    session.clear()
    return redirect('/login/')
  def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.get_id()

  def login(self):

    user = db.users.find_one({
      "email": request.form.get('email')
    })

    if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
      return self.start_session(user)
    
    return jsonify({ "error": "Invalid login credentials" }), 401

class Item(object):
  def __init__(self):
    self.collection_name = 'item'  # collection name

    self.fields = {
      "_id": uuid.uuid4().hex,
      "name": "string",
      "color": "string",
      "created": "datetime",
      "updated": "datetime",
    }

class Wardrobe:

  def isOK(self):

    return jsonify(self), 200

  def add(self):
    print(request.form)

    wardrobe = {
        "_id": uuid.uuid4().hex,
        "label": request.form.get('label'),
        "color": request.form.get('color'),
        "category": request.form.get('category'),
        "userId":  session['user']['_id']
      }


    if db.wardrobe.insert_one(wardrobe):
      return 'OK'