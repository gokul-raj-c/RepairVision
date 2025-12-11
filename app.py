# Import modules
from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

# Routes
@app.route('/')
def home():
    return render_template('index.html')   

@app.route('/upload')
def upload():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
