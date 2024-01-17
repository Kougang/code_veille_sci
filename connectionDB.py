import cherrypy
import mysql.connector
import os

# Configuration de la connexion à la base de données
cnx = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='dataset'
)
cursor = cnx.cursor()

