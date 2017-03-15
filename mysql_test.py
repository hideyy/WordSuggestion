# -*- coding: utf-8 -*-
"""
before this, you have to install mysql-connector-python
"""

import mysql.connector

#user information
USER_NAME = 'root'
PASSWORD = 'arai0806'
#connection
conn = mysql.connector.connect(user=USER_NAME, password=PASSWORD, host='localhost', database='test')
cur = conn.cursor()

#SQL文
cur.execute("select * from user;")


for row in cur.fetchall():
    print(row[0],row[1])

cur.close
conn.close
