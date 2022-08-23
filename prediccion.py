# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:32:49 2021

@author: maryf
"""

from ttkbootstrap import Style
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog, StringVar
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import math
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

matplotlib.use('TkAgg')


longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

####################################
###  FUNCIONES DE RED NEURONAL   ###
####################################
def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    res="Prediccion: Elefante"
  elif answer == 1:
    res="Prediccion: Rinoceronte"
  elif answer == 2:
    res="Prediccion: Zebra"
  print(res)

  return res
####################################
###     FUNCIONES DE FILTROS     ###
####################################

#--------CONTORNO-----------------#
def contorno(x , y):
    global path
    global currentImg
    if path != "":
        img=Image.open(path)
        img=img.resize((200,200))
        width, height = img.size
        data=np.array(img)/255
        dataFilter=np.zeros(data.shape)
        
        kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        R = conv(data[:, :, 0], kernel)
        G = conv(data[:, :, 1], kernel)
        B = conv(data[:, :, 2], kernel)
        dataFilter = abs(np.stack((R, G, B), axis=2) )
        imgFilter=Image.fromarray(np.uint8((dataFilter*255)))
        currentImg=imgFilter
        colocarImagen(x, y, imgFilter)
    else:
        messagebox.showinfo(message="Selecciona una imagen primero", title="Advertencia")
#--------LAPLACIANO-----------------#
def laplaciano(x , y):
    global currentImg
    global path
    if path != "":
        img=Image.open(path)
        img=img.resize((200,200))
        width, height = img.size
        data=np.array(img)/255
        dataFilter=np.zeros(data.shape)
        
        kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        R = conv(data[:, :, 0], kernel)
        G = conv(data[:, :, 1], kernel)
        B = conv(data[:, :, 2], kernel)
        dataFilter = abs(np.stack((R, G, B), axis=2) )
        imgFilter=Image.fromarray(np.uint8((dataFilter*255)))
        currentImg=imgFilter
        colocarImagen(x, y, imgFilter)
    else:
        messagebox.showinfo(message="Selecciona una imagen primero", title="Advertencia")
#--------SUAVIZADO-----------------#
def suavizado(x , y):
    global path
    global currentImg
    if path != "":
        img=Image.open(path)
        img=img.resize((200,200))
        width, height = img.size
        data=np.array(img)/255
        dataFilter=np.zeros(data.shape)
        for i in range(height-1):
            for j in range(width-1):
                dataFilter[i,j]=(data[i-1,j-1]+data[i-1,j]+data[i-1,j+1]+
                                 data[i,j-1]+data[i,j]+data[i,j+1]+
                                 data[i+1,j-1]+data[i+1,j]+data[i+1,j+1])/9

        imgFilter=Image.fromarray(np.uint8((dataFilter*255)))
        currentImg=imgFilter
        colocarImagen(x, y, imgFilter)
        
    else:
        messagebox.showinfo(message="Selecciona una imagen primero", title="Advertencia") 
#--------GAUSIANNO-----------------#
def gausianno(x , y):
    global path
    global currentImg
    if path != "":
        img=Image.open(path)
        img=img.resize((200,200))
        width, height = img.size
        data=np.array(img)/255
        dataFilter=np.zeros(data.shape)
        
        kernel=np.array([[1,2,1],[2,4,2],[1,2,1]])
        R = conv(data[:, :, 0], kernel)/16
        G = conv(data[:, :, 1], kernel/16)
        B = conv(data[:, :, 2], kernel)/16
        dataFilter = abs(np.stack((R, G, B), axis=2) )
        imgFilter=Image.fromarray(np.uint8((dataFilter*255)))
        currentImg=imgFilter
        colocarImagen(x, y, imgFilter)
        
    else:
        messagebox.showinfo(message="Selecciona una imagen primero", title="Advertencia") 
#--------FOURIER-----------------#
def transformadaFourier(x , y):
    global path
    global currentImg, arrayFourier
    if path != "":
        img=Image.open(path)
        filas, columnas = img.size
        imgRGB = img.convert("RGB")
        data=np.zeros(img.size,dtype=complex)
        print(data.shape)
        for i in range(filas):
            for j in range(columnas):
                r,g,b = imgRGB.getpixel((i,j))
                gris= int(r*0.299)+int(g*0.587)+int(b*0.114)
                data[i,j] = gris
        arrayFourier=np.arange(filas, columnas)
       #obtenemos la tranformada ayudandonos de  la1  propiedad de saparabilidad          
        for f in range(filas):
            data[f]=DFT(data[f])
            #arrayFourier[f]=data[f]
        for c in range(columnas):
            data[:,c]=DFT(data[:,c])
            #arrayFourier[:,c]=data[:,c]
        for i in range(filas):
            for j in range(columnas):
                pixelAbs=int(abs(data[i,j]))
                pixelDFT = tuple([pixelAbs, pixelAbs, pixelAbs])
                img.putpixel((i,j),pixelDFT)
        currentImg=img
        arrayFourier=data
        colocarImagen(x, y,img)
def filtradoFourier(x , y):
    global path
    global currentImg, arrayFourier
    if path != "":
        img=currentImg
        filas, columnas = img.size
        imgRGB = img.convert("RGB")
        data=np.zeros(img.size,dtype=complex)
        kernel=np.zeros(img.size,dtype=complex)
        
        for i in range(filas):
            for j in range(columnas):
                #llenamos el kernel
                distIJ=np.sqrt((i*i)+(j*j))
                const=int(filas*0.40)
                if i>((filas/2)-const) and i<((filas/2)+const) and j>((columnas/2)-const) and j<((columnas/2)+const) :
                    kernel[i,j]=0
                else:
                    kernel[i,j]=1       
        
        data=np.multiply(arrayFourier,kernel)
        for f in range(filas):
            data[f]=IDFT(data[f])
        for c in range(columnas):
            data[:,c]=IDFT(data[:,c])
       
        for i in range(filas):
            for j in range(columnas):
                pixelAbs=int(abs(data[i,j]))
                pixelDFT = tuple([pixelAbs, pixelAbs, pixelAbs])
                img.putpixel((i,j),pixelDFT)
        currentImg=img
        colocarImagen(x, y,img)
#--------HISTOGRAMA-----------------#
def histograma():
    global currentImg
    grises = np.zeros(256)
    global path
    if path != "":
        img=currentImg
        
        width, height = img.size
        imgRGB = img.convert("RGB")
        for i in range(width):
            for j in range(height):
                r,g,b = imgRGB.getpixel((i,j))
                gris= int(r*0.299)+int(g*0.587)+int(b*0.114)
                grises[gris] = grises[gris]+1
               
                pixelGray = tuple([gris, gris, gris])
                img.putpixel((i,j),pixelGray)
        dibujarHistograma(grises)
    else:
        messagebox.showinfo(message="Selecciona una imagen primero", title="Advertencia")
#--------FILTRO NEGATIVO-----------------#
def negativo(x , y):
    global path
    global currentImg
    if path != "":
        img=Image.open(path)
        width, height = img.size
        imgRGB = img.convert("RGB")
        for i in range(width):
            for j in range(height):
                r,g,b = imgRGB.getpixel((i,j))
                gris= int((r+g+b)/3)

                pixelGray = tuple([255-gris, 255-gris, 255-gris])
                img.putpixel((i,j),pixelGray)
        currentImg=img
        colocarImagen(x, y, img)
    else:
        messagebox.showinfo(message="Selecciona una imagen primero", title="Advertencia")
####################################
###     FUNCIONES AUXILIARES     ###
####################################   
def DFT(x):
#Calcula la tranformada en una dimension
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X    
def IDFT(x):
#Calcula la tranformada en una dimension
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    X= X/N
    return X   
def conv(img, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = img.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i,j] = np.sum(img[i:i+m, j:j+m]*kernel) 
        return new_image
    
def dibujarHistograma(arreglo):
    global x
    yHist = arreglo.astype(int)
    xHist = np.arange(0,256,1)

    fig = Figure(figsize=(2,2))
    a = fig.add_subplot(111)
    a.bar(xHist,yHist,color='#df691a')
   
    canvas = FigureCanvasTkAgg(fig, master=window)
    if x==230:
        canvas.get_tk_widget().place(x=860, y=340)
    else:
        canvas.get_tk_widget().place(x=x-210, y=340)
    canvas.draw()
def limpiarVentana():
    global x
    x=230
    canvas = Canvas(width=1080, 
                    height=600, 
                    bd=0, 
                    highlightthickness=0, 
                    relief='ridge', 
                    bg='#2b3e50')
    canvas.place(x=0, y=100)
def opcionesMenu(tipo):
    global x, y 
    if tipo=="Detectar bordes ":
        contorno(x,y)
    elif tipo=="Laplaciano         ":
       laplaciano(x,y)
    elif tipo=="Suavizado lineal" :
        suavizado(x,y)
    elif tipo=="Gaussiano          ": 
        gausianno(x,y)
    elif tipo=="Fourier              ": 
        transformadaFourier(x,y)
    elif tipo=="Filtrado fourier     ":
        filtradoFourier(x,y)
    elif tipo=="Negativo             ":
        negativo(x,y)
    label=Label(window, text=tipo)
    label.config(fg="white" )
    label.place(x=x+50, y=303)
    if x==860:
        x=230
    else:
        x=x+210
   
####################################
###     FUNCIONES CARGA IMG      ###
####################################  
def cargarImagen():
    imagen_cargada=filedialog.askopenfilename(title="Selecciona imagen",
                                              filetypes=(("JPG", "*jpg"),("PNG", "*.png"),
                                                         ("JPEG", "*.jpeg"), 
                                                         ("GIF", "*.gif"),
                                                         
                                                         ))
    print("\n Path Imagen ",imagen_cargada)
    return imagen_cargada

def mostrarImagen():
    global path, labelText
    #limpiarVentana()
    path=cargarImagen()
    img=Image.open(path)
    colocarImagen(20, 100, img)
    tipo=predict(path);
    
    labelText.set(predict(path))
    
def colocarImagen(xPos , yPos, img):
    global labelText;
    
    img=img.resize((200,200), Image.ANTIALIAS)
    img=ImageTk.PhotoImage(img)
    panel1=ttk.Label(window, image=img)
    panel1.place(x=xPos, y=yPos)
    panel1.image=img
    canvas = Canvas(width=200, 
                    height=200, 
                    bd=0, 
                    highlightthickness=0, 
                    relief='ridge', 
                    bg='#2b3e50')
    canvas.place(x=xPos, y=340)
    labelTipo=ttk.Label(window,  textvariable=labelText,
                 font="Helvetica 12").place(x=30, y=350)

def display_selected(choice):
    choice = lista.get()
    print(choice)  
    opcionesMenu(choice)    

style = Style(theme="superhero")

window = style.master
window.title("Filtros en imagenes ")
window.geometry("1080x600")

path = ""
currentImg = Image
arrayFourier=np.arange(0)
x=230
y=100
contList= 0

OptionList = ["Seleccionar filtro",
              "Detectar bordes ",
              "Laplaciano         ",
              "Suavizado lineal",
              "Gaussiano          ",
              "Fourier              ",
              "Filtrado fourier     ",
              "Negativo             "] 

lista = StringVar()
labelText = StringVar()
lista.set(OptionList[3])
window.titulo=ttk.Label(window,  text = "Filtros de imágenes",
                 font="Helvetica 25").pack(side="top")
boton=ttk.Button(window, text="Cargar imagen", style='warning.TButton',
                 command=mostrarImagen)
boton.config(width=20)
boton.place(x=40, y=60)

opt = ttk.OptionMenu(window, lista, *OptionList, command=display_selected)
opt.config(width=17)
opt.place(x=230, y=60)

botonHistograma=ttk.Button(window, text="Histograma", style='primary.TButton',
                 command=histograma)
botonHistograma.config(width=20)
botonHistograma.place(x=420, y=60)

botonLimpiar=ttk.Button(window, text="Reset", style='primary.TButton',
                 command=limpiarVentana)
botonLimpiar.config(width=20)
botonLimpiar.place(x=630, y=60)
window.mainloop()


