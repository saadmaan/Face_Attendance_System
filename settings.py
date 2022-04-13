from flask import Flask, request, jsonify, make_response, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import enum 
import hashlib
import json
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import jwt as jwt
import argparse
from functools import wraps


app = Flask(__name__)

app.config['SECRET_KEY']='Th1s1ss3cr3t'
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database2.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
