from flaskapp import app
from flaskapp.user.models import User
from flaskapp.user.models import Wardrobe

@app.route('/user/signup', methods=['POST'])
def signup():
  return User().signup()

@app.route('/user/signout')
def signout():
  return User().signout()

@app.route('/user/login', methods=['POST'])
def login():
  return User().login()

@app.route('/wardrobe/add', methods=['POST','GET'])
def add():
  return Wardrobe().add()