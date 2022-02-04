# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:39:57 2022

@author: Martin
"""

class Auto:
    
    def __init__(self, color, marca, precio, ruedas = 4, kilometraje = 0):
        self.color = color
        self.ruedas = ruedas
        self.marca = marca
        self. precio = precio
        self.kilometraje = kilometraje
        
    def addKms(self, kms):
        self.kilometraje = self.kilometraje + kms
            
    def setKilometraje(self, kms):
        self.kilometraje = kms
    
    def getKilometraje(self):
        return self.kilometraje
    
    def calcNaftaTotal(self):
        return self.kilometraje * 10
    
    def getColor(self):
        return self.color
    
miAuto=Auto(0,0,0)
miAuto.getKilometraje()
miAuto=Auto(12,6,"BMW")
miAuto.getColor()
