#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc


# In[2]:
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


# In[3]:


from tensorflow.keras.models import load_model


# In[5]:


model = load_model('modelo_2d_lineas1.h5')


# In[6]:


model1 = load_model('modelo1D_3.h5')


# In[7]:


datos = pd.read_csv("iex_dow.csv")

datos.index = datos.date
datos = datos.drop('date', axis=1)


# In[8]:


datos.columns = ["open", "high", "low", "close","ticker"]


# In[9]:


datos.head()


# In[10]:


ticker_datos = datos.ticker.unique()
ticker_datos


# In[11]:


len(ticker_datos)


# Definimos la función para plotear

# In[12]:


def plot_candel(data):
    fig, ax = plt.subplots(figsize=(4,3))
    _ = candlestick2_ohlc(ax, data.open, data.high,
                          data.low, data.close,
                          colorup='g', colordown='r', width=0.66, alpha=0.8)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    return plt.show()


# Bucle para generar fotos

# In[13]:


ticker_datos[13]


# In[14]:


import math
from scipy.signal import savgol_filter


# In[15]:


def plot_candel1(data,nombre):
    fig, ax = plt.subplots(figsize=(4,3))
    media_pintar = data.iloc[:,0:4].mean(axis=1)
    media_pintar_smooth = savgol_filter(media_pintar, 5, 3)
    media_pintar_smooth = pd.Series(media_pintar_smooth)
    media_pintar_smooth.index = np.arange(0,media_pintar_smooth.shape[0])
    
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    
    plt.plot(media_pintar_smooth, color='red',linewidth=3)
    plt.savefig(nombre,bbox_inches='tight')
    return plt.close(fig)


# ### Todos los días vamos a hacer 3 fotos por ticker, una a 30 días, otra a 20 y otra a 15

# In[16]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-30):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 30
    datos_grafico = datos_activo
    nombre = "graficos_hoy_lineas/"+ticker_datos[activo]+"30.png"
    plot_candel1(datos_grafico,nombre)


# In[17]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-20):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 20
    datos_grafico = datos_activo
    nombre = "graficos_hoy_lineas/"+ticker_datos[activo]+"20.png"
    plot_candel1(datos_grafico,nombre)


# In[18]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-15):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 15
    datos_grafico = datos_activo
    nombre = "graficos_hoy_lineas/"+ticker_datos[activo]+"15.png"
    plot_candel1(datos_grafico,nombre)


# In[19]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-40):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 15
    datos_grafico = datos_activo
    nombre = "graficos_hoy_lineas/"+ticker_datos[activo]+"40.png"
    plot_candel1(datos_grafico,nombre)


# In[20]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-50):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 15
    datos_grafico = datos_activo
    nombre = "graficos_hoy_lineas/"+ticker_datos[activo]+"50.png"
    plot_candel1(datos_grafico,nombre)


# In[ ]:





# In[21]:


def plot_candel1(data,nombre):
    fig, ax = plt.subplots(figsize=(4,3))
    _ = candlestick2_ohlc(ax, data.open, data.high,
                          data.low, data.close,
                          colorup='g', colordown='r', width=0.66, alpha=0.8)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.savefig(nombre,bbox_inches='tight')
    return plt.close(fig)


# In[22]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-30):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 30
    datos_grafico = datos_activo
    nombre = "graficos_hoy/"+ticker_datos[activo]+"30.png"
    plot_candel1(datos_grafico,nombre)


# In[23]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-20):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 20
    datos_grafico = datos_activo
    nombre = "graficos_hoy/"+ticker_datos[activo]+"20.png"
    plot_candel1(datos_grafico,nombre)


# In[24]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-15):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 15
    datos_grafico = datos_activo
    nombre = "graficos_hoy/"+ticker_datos[activo]+"15.png"
    plot_candel1(datos_grafico,nombre)


# In[25]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-40):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 15
    datos_grafico = datos_activo
    nombre = "graficos_hoy/"+ticker_datos[activo]+"40.png"
    plot_candel1(datos_grafico,nombre)


# In[26]:


for activo in np.arange(0,len(ticker_datos)):
    datos_activo = datos[datos.ticker==ticker_datos[activo]]
    datos_activo = datos_activo.iloc[(datos_activo.shape[0]-50):datos_activo.shape[0],:] #asegurarnos de que estamos cogiendo 15
    datos_grafico = datos_activo
    nombre = "graficos_hoy/"+ticker_datos[activo]+"50.png"
    plot_candel1(datos_grafico,nombre)


# In[ ]:





# In[ ]:





# In[ ]:





# LLamamos a esos gráficos

# In[27]:


import cv2


# In[28]:


datos_guardados_graficos = []
nombres_fotos = []
for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy_lineas/"+ticker_datos[figura]+"30.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos.append(image)
    nombres_fotos.append(nombre)


# In[29]:


for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy_lineas/"+ticker_datos[figura]+"20.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos.append(image)
    nombres_fotos.append(nombre)


# In[30]:


for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy_lineas/"+ticker_datos[figura]+"15.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos.append(image)
    nombres_fotos.append(nombre)


# In[31]:


for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy_lineas/"+ticker_datos[figura]+"40.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos.append(image)
    nombres_fotos.append(nombre)





for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy_lineas/"+ticker_datos[figura]+"50.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos.append(image)
    nombres_fotos.append(nombre)


# In[33]:


nombres_fotos[0:5]





datos_guardados_graficos1 = []
nombres_fotos1 = []
for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy/"+ticker_datos[figura]+"30.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos1.append(image)
    nombres_fotos1.append(nombre)





for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy/"+ticker_datos[figura]+"20.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos1.append(image)
    nombres_fotos1.append(nombre)





for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy/"+ticker_datos[figura]+"15.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos1.append(image)
    nombres_fotos1.append(nombre)





for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy/"+ticker_datos[figura]+"40.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos1.append(image)
    nombres_fotos1.append(nombre)




for figura in np.arange(0,len(ticker_datos)):
    nombre =  "graficos_hoy/"+ticker_datos[figura]+"50.png"

    image = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE) #GRAYSCALE

    datos_guardados_graficos1.append(image)
    nombres_fotos1.append(nombre)




datos_guardados1 = np.array(datos_guardados_graficos1)
predicciones_hechas = pd.read_csv('predicciones.csv')

predicciones_hechas = predicciones_hechas.drop('Unnamed: 0', axis = 1)

nombres_limpios = []
for names in np.arange(0,len(nombres_fotos)):
    name = nombres_fotos[names].replace("graficos_hoy/","").replace(".png", "")
    nombres_limpios.append(name)

nombres_limpios = []
for names in np.arange(0,len(nombres_fotos)):
    name = nombres_fotos[names].replace("graficos_hoy_lineas/","").replace(".png", "")
    nombres_limpios.append(name)

nombres_limpios = np.array(nombres_limpios)
numero_por_ventana = int(len(nombres_limpios)/5) #OJO SI PLOTEAS CON VENTANAS DE 50 LO QUE TIENES SON 4 NO 3

x1 = np.full(numero_por_ventana,30)
x2 = np.full(numero_por_ventana,20)
x3 = np.full(numero_por_ventana,15)
x4 = np.full(numero_por_ventana,40)
x5 = np.full(numero_por_ventana,50)

ventana = pd.concat([pd.Series(x1), pd.Series(x2), pd.Series(x3), pd.Series(x4), pd.Series(x5)])
ventana.index = np.arange(0,ventana.shape[0])

def limpia_nombre(nombres_series):
    lista_nombres = []
    for nombres in np.arange(0,len(nombres_series)):
        nombre = nombres_series[nombres]
        nombre = nombre[0:len(nombre)-2]
        lista_nombres.append(nombre) 
   
    return pd.Series(lista_nombres)


def nombre_figura(predicciones0,predicciones1,predicciones2,predicciones3,predicciones4):
    cero = np.repeat("Hombro c hombro inverso",len(predicciones0))
    uno = np.repeat("Hombro c hombro",len(predicciones1))
    dos = np.repeat("Doble techo",len(predicciones2))
    tres = np.repeat("Sin grafico",len(predicciones3))
    cuatro = np.repeat("Doble suelo",len(predicciones4))
    nombres = pd.concat([pd.Series(cero), pd.Series(uno), pd.Series(dos), pd.Series(tres), pd.Series(cuatro)])
    return nombres


confianza=0.9

predicciones0 = pd.Series(predicciones_hechas.iloc[:,0])
predicciones0 = predicciones0[predicciones0>confianza]
    
predicciones1 = pd.Series(predicciones_hechas.iloc[:,1])
predicciones1 = predicciones1[predicciones1>confianza]
    
predicciones2 = pd.Series(predicciones_hechas.iloc[:,2])
predicciones2 = predicciones2[predicciones2>confianza]
    
predicciones3 = pd.Series(predicciones_hechas.iloc[:,3])
predicciones3 = predicciones3[predicciones3>confianza]
    
predicciones4 = pd.Series(predicciones_hechas.iloc[:,4])
predicciones4 = predicciones4[predicciones4>confianza] #tiene mas dobles suelos vistos, los identifica mejor
    
predicciones_totales = predicciones0.append(predicciones1).append(predicciones2).append(predicciones3).append(predicciones4)
numero_serie = predicciones_totales.index

activos = limpia_nombre(nombres_limpios[numero_serie])
print(activos)
ventana1 = ventana[numero_serie]
ventana1.index = np.arange(0,ventana1.shape[0])
figura = nombre_figura(predicciones0,predicciones1,predicciones2,predicciones3,predicciones4)
figura.index = np.arange(0,figura.shape[0])
probabilidad = predicciones_totales
probabilidad.index = np.arange(0,probabilidad.shape[0])
        
data = {'Activo': activos, 'Ventana_dias': ventana1, 'figura_encontrada':figura,'probabilidad':probabilidad}
df = pd.DataFrame(data)
df1 = df[df.figura_encontrada != "Sin grafico"]
df1['Indice'] = range(df1.shape[0])
df1 = df1[['Indice','Activo','Ventana_dias','figura_encontrada','probabilidad']]

import dash
import dash_table
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


data_to_plot = datos[datos.ticker == 'AAPL']


fig = go.Figure(
        go.Candlestick(
            #x=data_to_plot['Date'],
            open=data_to_plot['open'],
            high=data_to_plot['high'],
            low=data_to_plot['low'],
            close=data_to_plot['close']
        )
    )
fig.update_layout(xaxis_rangeslider_visible=False)

conf = (0.5,0.6,0.7,0.8,0.9)
ventanas = (15,20,30,40,50)


app.layout = html.Div([
    html.Div(
            children=[
                html.H1(children="ANÁLISIS DE PATRONES", className="header-title", style= {'color': 'blue', 'font-size': '300', 'text-align':'center'}),
            ],
        className="header",
        ),
    html.Div([
        html.Div([
            dcc.Graph(
                id = 'velas',
                figure = fig,
                config = {
                'displayModeBar': False})
    ], style={'width': '70%','display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),

    
    html.Div([
        dcc.Input(id="num_grafico", type="number", placeholder="Numero de grafico", style={'display':'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
        html.Button('Resumen', id='resumen_bot', n_clicks=0, style={'display':'block', 'margin-left': 'auto', 'margin-right': 'auto'})]),

        html.Div([
            dash_table.DataTable(
                id = 'dataframe',
                columns=[{"name": i, "id": i} for i in df1.columns],
                data=df1.to_dict('records'),

                style_cell={'textAlign': 'center'},
                
                style_data={
                    'color': 'black',
                    'backgroundColor': 'white'
                    },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(140, 237, 255)',
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(0, 203, 255)',
                    'color': 'black',
                    'fontWeight': 'bold'
                    }
                ),
    ],style={'width': '100%', 'display': 'inline-block'}),

    ])
])


@app.callback(
   Output('velas', 'figure'), # src 
   Input('num_grafico', 'value'))
   #Input('confianza', 'value'))

def update_figure(num_grafico):

    datos_sacado = df1.iloc[num_grafico,:]
    ticker = datos_sacado.Activo
    dato_ventana = datos_sacado.Ventana_dias
    
    activo_sacado = datos[datos.ticker==ticker]
    data_to_plot = activo_sacado.iloc[(activo_sacado.shape[0]-dato_ventana):activo_sacado.shape[0],:]
    fig = go.Figure(
        go.Candlestick(
            #x=data_to_plot['Date'],
            open=data_to_plot['open'],
            high=data_to_plot['high'],
            low=data_to_plot['low'],
            close=data_to_plot['close']
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig

@app.callback(
   Output('dataframe', 'data'), 
   Input('num_grafico', 'value'),
   Input('resumen_bot', 'n_clicks'))

def update_table(num_grafico, resumen_bot):
    datos_tabla = df1.iloc[[num_grafico]]
    datos_tabla = datos_tabla.to_dict('records')
    if resumen_bot > 0:
        resumen_bot = 0
        return df1.to_dict('records')
    else:
        return datos_tabla

print('FIN')

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False, port=8080)





# %%

!pip list
# %%