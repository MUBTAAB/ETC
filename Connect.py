# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:33:59 2017

@author: mucs_b
"""

import cx_Oracle

con = cx_Oracle.connect()
print(con.version)

con.close()


