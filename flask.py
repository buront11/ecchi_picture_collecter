from flask import Flask
app = Flask(__name__)

@app.route('/')
def ecchi_picture_collecter():
    name = "Hello world"
    return name

if __name__=='__main__':
    app.run(debug=True)