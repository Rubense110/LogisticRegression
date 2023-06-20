#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Villanueva Duque
# NOMBRE: Hugo
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: Jiménez Jiménez
# NOMBRE: Rubén
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

from carga_datos import *
import numpy as np

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------
from carga_datos import *
import numpy as np

def particion_entr_prueba(X, y, test=0.2):
    
    # Inicializamos los índices vacíos
    indices_test = list()
    indices_entr = list()

    # haremos la división por cada clase, y lo iremos añadiendo a la final
    for clase in np.unique(y):

        indicesClase = np.where(y==clase)[0]    # Cogemos los índices de la clase actual
        np.random.shuffle(indicesClase)         # al hacer un shuffle forzamos que sea aleatorio
        tam_test = int(len(indicesClase)*test)  # escogemos cuantos seran para test

        indices_test.extend(indicesClase[:tam_test]) # Finalmente con slicing escogemos los que seran
        indices_entr.extend(indicesClase[tam_test:]) # para entr/test de la clase y lo añadimos al total

    # tenemos los índices, ahora escogemos los ejemplos del Dataset Original
    X_entr = X[indices_entr]
    y_entr = y[indices_entr]
    X_test = X[indices_test]
    y_test = y[indices_test]

    return X_entr, X_test, y_entr, y_test

## ---------- 

# Test 

Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

def testEj1():

    print("\n###########################################")
    print("############### EJERCICIO 1 ###############")
    print("###########################################\n")

    clases, count = np.unique(y_votos,return_counts=True)
    clasesE, countE = np.unique(ye_votos,return_counts=True)
    clasesP, countP = np.unique(yp_votos,return_counts=True)

    print("->              Datos de Votos\n")
    print("   Proporciones      - Original: {0} - Entrenamiento: {1} - Prueba: {2}".format(y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]), "\n")
    print("   Conjunto Original - clases:{0} - Nº de ejemplos: {1} - Distribución: {2}    ".format(clases, count, (count / np.sum(count)) * 100))
    print("   Entrenamiento     - clases:{0} - Nº de ejemplos: {1} - Distribución: {2}    ".format(clasesE, countE, (countE / np.sum(countE)) * 100))
    print("   Prueba            - clases:{0} - Nº de ejemplos: {1} - Distribución: {2}    ".format(clasesP, countP, (countP / np.sum(countP)) * 100),"\n")

    clases, count = np.unique(y_cancer,return_counts=True)
    clasesEv, countEv = np.unique(yev_cancer,return_counts=True)
    clasesP, countP = np.unique(yp_cancer,return_counts=True)
    clasesE, countE = np.unique(ye_cancer,return_counts=True)
    clasesV, countV = np.unique(yv_cancer,return_counts=True)


    print("->              Datos de Cancer\n")
    print("   Proporciones      - Original: {0} - Entrenamiento: {1}    - Prueba: {2}".format(y_cancer.shape[0],ye_cancer.shape[0],yp_cancer.shape[0]), "\n")
    print("   Conjunto Original - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clases, count, (count / np.sum(count)) * 100))
    print("   Entr y Val        - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clasesEv, countEv, (countEv / np.sum(countEv)) * 100))
    print("   · Entrenamiento   - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clasesE, countE, (countE / np.sum(countE)) * 100))
    print("   · Validacion      - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clasesV, countV, (countE / np.sum(countE)) * 100))
    print("   Prueba            - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clasesP, countP, (countP / np.sum(countP)) * 100),"\n")


    clases, count = np.unique(y_credito,return_counts=True)
    clasesE, countE = np.unique(ye_credito,return_counts=True)
    clasesP, countP = np.unique(yp_credito,return_counts=True)

    print("->              Datos de Creditos\n")
    print("   Proporciones      - Original: {0} - Entrenamiento: {1}    - Prueba: {2}".format(y_cancer.shape[0],ye_cancer.shape[0],yp_cancer.shape[0]), "\n")
    print("   Conjunto Original - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clases, count, (count / np.sum(count)) * 100))
    print("   Entrenamiento     - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clasesE, countE, (countE / np.sum(countE)) * 100))
    print("   Prueba            - clases:{0}    - Nº de ejemplos: {1}   - Distribución: {2}    ".format(clasesP, countP, (countP / np.sum(countP)) * 100),"\n")


# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


# class NormalizadorStandard():

#    def __init__(self):

#         .....
        
#     def ajusta(self,X):

#         .....        

#     def normaliza(self,X):

#         ......

# 


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


# Por ejemplo:
    
    
# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 



# ------ 

class NormalizadorStandard():
    
    def __init__(self):
        self.ajustado = False

    # Ajusta calcula los valores para cada valor de X
    def ajusta(self, X):
        self.media = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.ajustado = True    # Ahora está ajustado

    # Normalizamos los datos
    def normaliza(self, X):
        if not self.ajustado:   # Si no pasamos por ajusta debemos hacer que salte error
            raise NormalizadorNoAjustado("Normalizador no ajustado")
        return (X - self.media) / self.std
    
# ----------
# Test

normst_cancer=NormalizadorStandard()
normst_cancer.ajusta(Xe_cancer)
Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

X = np.array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])

normalizador = NormalizadorStandard()
normalizador.ajusta(X)
X_normalizado = normalizador.normaliza(X)

def testEj2_1():

    print("\n###########################################")
    print("############## EJERCICIO 2.1 ##############")
    print("###########################################\n")

    print("##->              Datos de Cancer Normalizados Standard\n")
    print("Entrenamiento\n\n- Media   {0}\n- Desviaciación típica   {1}\n".format(np.mean(Xe_cancer_n), np.std(Xe_cancer_n, axis = 0)))
    print("Validación   \n\n- Media   {0}\n- Desviaciación típica   {1}\n".format(np.mean(Xv_cancer_n), np.std(Xv_cancer_n, axis = 0)))
    print("Prueba       \n\n- Media   {0}\n- Desviaciación típica   {1}\n".format(np.mean(Xp_cancer_n), np.std(Xp_cancer_n, axis = 0)))

    print("##->              Datos de Prueba\n")
    print("Datos Originales  :\n {}\n".format(X))
    print("Datos Normalizados:\n {}\n".format(X_normalizado))
    print("- Media   {0}\n- Desviaciación típica   {1}\n".format(np.mean(X_normalizado), np.std(X_normalizado, axis = 0)))





# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 


# ------ 
class NormalizadorMinMax:
    def __init__(self):
        self.ajustado = False

    # Calcula el mínimo y el máximo de cada Característica (axis=0) de X
    def ajusta(self, X):
        self.minimo = np.min(X, axis=0)
        self.maximo = np.max(X, axis=0)
        self.ajustado = True    # Ahora está ajustado

    # Normalizamos los datos
    def normaliza(self, X):
        if not self.ajustado:
            raise NormalizadorNoAjustado("Normalizador no ajustado")    # Si no pasamos por ajusta debemos hacer que salte error
        return (X - self.minimo) / (self.maximo - self.minimo)

# ----------
# Test

normst_cancer_minmax=NormalizadorMinMax()
normst_cancer_minmax.ajusta(Xe_cancer)
Xe_cancer_n_minmax=normst_cancer_minmax.normaliza(Xe_cancer)
Xv_cancer_n_minmax=normst_cancer_minmax.normaliza(Xv_cancer)
Xp_cancer_n_minmax=normst_cancer_minmax.normaliza(Xp_cancer)

normalizador_minmax = NormalizadorMinMax()
normalizador_minmax.ajusta(X)
X_normalizado_minmax = normalizador_minmax.normaliza(X)

def testEj2_2():

    print("\n###########################################")
    print("############## EJERCICIO 2.2 ##############")
    print("###########################################\n")

    print("##->              Datos de Cancer Normalizados MinMax\n")
    print("- Datos Originales  :\n {}\n".format(Xe_cancer))
    print("- Datos Normalizados:\n {}\n".format(Xe_cancer_n_minmax))
    print("- Mínimo: ",np.min(Xe_cancer_n_minmax))
    print("- Máximo: {}\n".format(np.max(Xe_cancer_n_minmax)))


    print("##->              Datos de Prueba\n")
    print("- Datos Originales  :\n {}\n".format(X))
    print("- Datos Normalizados:\n {}\n".format(X_normalizado_minmax))












# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass


# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit    
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])



# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------
from scipy.special import expit    

#Func proporcionada 
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

from scipy.special import expit    

def sigmoide(x):
    return expit(x)


class RegresionLogisticaMiniBatch:
    def __init__(self, rate=0.1, rate_decay=False, n_epochs=100, batch_tam=64, random_seed=1):
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.pesos = None
        self.seed = random_seed

    def entrena(self, X, y, Xv=None, yv=None, n_epochs=100, salida_epoch=False,
                 early_stopping=False, paciencia=3):
            # Las clases están dadas por los valores de y, positiva en 2º lugar como es definida en los datos
            self.clases = np.unique(y)
            num_caract = X.shape[1]
            self.pesos = np.random.randn(num_caract)

            mejor_EC = float('inf')
            cuenta_paciencia = 0
            # Calcular e imprimir EC y rendimiento iniciales
            if salida_epoch:
                
                EC_inicial = np.sum(self.entropia_cruzada(y,self.clasifica(X)))
                rend_inicial = rendimiento(self, X, y)
                print("\t-------")
                print(f"\tInicialmente, en entrenamiento EC: {EC_inicial}, rendimiento: {rend_inicial}\n")

            for epoch in range(n_epochs):
                 # Barajar los datos al inicio de cada época
                idx = np.random.permutation(X.shape[0])
                X = X[idx]
                y = y[idx]
                # De 0 a tam_X, saltando de batch en batch
                for i in range(0, X.shape[0], self.batch_tam):
                    X_mini, y_mini = X[i:i+self.batch_tam], y[i:i+self.batch_tam]
                    grad = np.dot(X_mini.T, (y_mini - self.clasifica_prob(X_mini) ))
                    
                    # Actualizacion de tasa de aprendizaje  
                    if self.rate_decay:
                        self.rate = self.rate * (1/(1 + epoch))

                    # Actualizacion de pesos
                    self.pesos += self.rate * grad

                if salida_epoch or early_stopping:
                    EC_train = np.sum(self.entropia_cruzada(y,self.clasifica(X)))
                    if salida_epoch:
                        rend = rendimiento(self,X, y)
                        print("\t-------")
                        print(f"\tEpoch {epoch}\n\t\ten entrenamiento EC: {EC_train}, rendimiento: {rend}")
                        if Xv is not None and yv is not None:
                            EC_val =   np.sum(self.entropia_cruzada(yv,self.clasifica(Xv)))
                            rend_val = rendimiento(self, Xv, yv)
                            print(f"\t\ten validación    EC: {EC_val}, rendimiento : {rend_val}\n")

                    if early_stopping:
                        
                        #No se usa validacion cuando hay early stopping: Xv e yv son resp. X e y.
                        Xv = X
                        yv = y
                        EC_val =  np.sum(self.entropia_cruzada(yv,self.clasifica(Xv)))
                        if EC_val < mejor_EC:
                            mejor_EC = EC_val
                            cuenta_paciencia = 0
                        else:
                            cuenta_paciencia += 1
                            if cuenta_paciencia >= paciencia:
                                if salida_epoch:
                                    print("\t-------")
                                    print("\tPARADA TEMPRANA\n")
                                    return

    def clasifica_prob(self, X):
        if self.pesos is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado.")
        z = np.dot(X, self.pesos)
        # expit(z) equivale a sigmoide(z)
        return sigmoide(z)
    

    def clasifica(self, X):
        return np.where(self.clasifica_prob(X) >= 0.5, self.clases[1], self.clases[0])
    
        
    def entropia_cruzada(self, y,y_pred):
        epsilon = 0.0000000001
        return (-y * np.log(y_pred + epsilon) - (1 - y) * np.log(1 - y_pred + epsilon))
    
def test_Ej3():

    lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True, random_seed=1)
    print("\n###########################################")
    print("############### EJERCICIO 3 ###############")
    print("###########################################\n")

    print("##->              Datos de Cancer\n")
    lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)
    predicciones = lr_cancer.clasifica(Xp_cancer_n[24:27])
    probabilidades = lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
    rendCEntren = rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
    rendCPrueba = rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)

    print("\t- Predicciones\n\t   Ejemplo 24: {0}\n\t   Ejemplo 25: {1}\n\t   Ejemplo 26: {2}\n ".format(predicciones[0],predicciones[1],predicciones[2]))
    print("\t- Valores Reales\n\t   Ejemplo 24: {0}\n\t   Ejemplo 25: {1}\n\t   Ejemplo 26: {2}\n ".format(yp_cancer[24:27][0],yp_cancer[24:27][1],yp_cancer[24:27][2]))
    print("\t- Probabilidad de pertenecer a la clase positiva:")
    print("\t   Ejemplo 24: {0}\n\t   Ejemplo 25: {1}\n\t   Ejemplo 26: {2}\n ".format(probabilidades[0],probabilidades[1],probabilidades[2]))
    print("\t- Rendimiento\n\t   Entrenamiento: {0}\n\t   Prueba: {1}".format(rendCEntren,rendCPrueba))


# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================



# Este ejercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328




# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------

def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, n=5, verbose=True):
    tam_X = X.shape[0]
    # Crear índices para la estratificación
    indices = np.arange(tam_X)
    np.random.shuffle(indices)
    # Tamaño de cada particion
    tam_part = tam_X // n
    rendimientos = []
    indices_test = list()
    indices_entr = list()

    # Iterar a través de cada partición para usarla como conjunto de prueba
    # Para hacer una division estratificada seguiremos el método que usamos en el ejercicio 1
    # Es decir, dividiremos para cada clase y luego combinaremos los subconjuntos 
    
    for i in range(n):
        for clase in np.unique(y):
            # Crear conjuntos de entrenamiento y prueba para esta iteración
            # Primero creamos el conjunto de prueba
            indices_clase = np.where(y==clase)[0]
            indices_test.extend(indices_clase[i*tam_part:(i+1)*tam_part]) 
            # Ahora excluimos los índices que están en ese conjunto de los que usaremos para entrenar
            indices_entr.extend(np.concatenate((indices_clase[:i*tam_part], indices_clase[(i+1)*tam_part:])))  

        X_entrena, X_test = X[indices_entr], X[indices_test]
        y_entrena, y_test = y[indices_entr], y[indices_test]

        # Crear y entrenar el modelo
        modelo = clase_clasificador(**params)
        modelo.entrena(X_entrena, y_entrena)

        # Evaluar el rendimiento y almacenarlo
        rend = rendimiento(modelo, X_test, y_test)
        rendimientos.append(rend)
        if verbose:
            print("\t-------")
            print(f"\tPartición: {i+1}. Rendimiento:{rend}")
        # Devolver el rendimiento medio
    if verbose: print("\t-------")
    return np.mean(rendimientos)

#------------------------------------------------------------------------------

# test

def testEj4():
    print("\n###########################################")
    print("############### EJERCICIO 4 ###############")
    print("###########################################\n")

    print("##->              Datos de Cancer\n")
    r = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,{"batch_tam":16,"rate":0.01,"rate_decay":True},Xe_cancer_n,ye_cancer,n=5)
    lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
    lr16.entrena(Xe_cancer_n,ye_cancer)

    print(f"\n\tResultado Medio:               {r}")
    print(f"\tRendimiento en Cjto de Prueba: {rendimiento(lr16,Xp_cancer_n,yp_cancer)}\n")




# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ----------------------------

# Vamos a definir una función que haga una simple búsqueda de parámetros, en este caso, hará
# una búsqueda exhaustiva pero de pocas combinaciones para no complicarlo demasiado.

# La función recibirá el Conjunto de datos entero, y como parámetros opcionales podemos especificar
# el tamaño de los conjuntos de validación y prueba, que por defecto son 0.2
# también podemos especificar el normalizador, por si queremos usar uno que no sea el Standard

def Ajusta_parametros(Xev,yev,Xp,yp,val=0.2):

    # Hacemos la separación de los datos, por no complicar el código, forzamos que haya que
    # especificar conjunto de validación 
    if val == 0 : raise Exception("Especifica tamaño de validación")
    Xe,Xv,ye,yv=particion_entr_prueba(Xev,yev,test=val)

    # Estos valores se irán actualizando para conservar los que den mayor rendimiento
    mejor_rendimiento = 0
    mejor_rate = 0
    mejor_batch_tam = 0
    mejor_rate_decay = False

    print("\n-------")
    print("Comenzando búsqueda exhaustiva de parámetros")
    print("-------\n")

    # Comenzamos la búsqueda
    for l_rate in [0.1, 0.01, 0.001]:
        for batch_tam in [16, 32, 64, 128]:
            for rate_decay in [True, False]:
                # Creamos y entrenamos el modelo
                lr = RegresionLogisticaMiniBatch(rate=l_rate, batch_tam=batch_tam, rate_decay=rate_decay)
                lr.entrena(Xe, ye, Xv, yv, salida_epoch=False,early_stopping=True)

                # Evaluamos su rendimiento
                rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,{"batch_tam":batch_tam,"rate":l_rate,"rate_decay":rate_decay}, Xe,ye,n=5, verbose=False)
                print(f" rate: {l_rate}, batch_tam: {batch_tam}, rate_decay: {rate_decay}, rendimiento: {rend} ")

                # Actualizamos mejores parámetros si es necesario
                if rend > mejor_rendimiento:
                    mejor_rendimiento = rend
                    mejor_rate = l_rate
                    mejor_batch_tam = batch_tam
                    mejor_rate_decay = rate_decay
    print("\n-------")
    print(f"Mejores parámetros encontrados: rate: {mejor_rate}, batch_tam: {mejor_batch_tam}, rate_decay: {mejor_rate_decay}, rendimiento: {mejor_rendimiento}")
    print("\nEntrenando con los mejores parámetros: ")
    print("-------\n")

    # Una vez tenemos los mejores Parámetros entrenaremos un modelo con estos, y daremos su rendimiento sobre el cjto de prueba
    lr_best = RegresionLogisticaMiniBatch(rate=mejor_rate, batch_tam=mejor_batch_tam, rate_decay=mejor_rate_decay)
    lr_best.entrena(Xe, ye, Xv ,yv , early_stopping=True, salida_epoch=True)

    print("\n-------")
    print(f"Rendimiento sobre Conjunto de Prueba: {rendimiento(lr_best, Xp, yp)}")
    print("-------\n")

def testEj5():
        print("\n###########################################")
        print("############### EJERCICIO 3 ###############")
        print("###########################################\n")

        # Ya tenemos estos datos separados y normalizados de ejercicios anteriores
        print("##->              Datos de Cancer\n")
        Ajusta_parametros(Xe_cancer_n, ye_cancer, Xp_cancer_n, yp_cancer)

        # regresión logística binaria necesita que los valores de clasif sean 0 y 1,
        #  hacemos la conversión con np.where()
        print("##->              Datos de Votos\n")
        y_votos_lr = np.where(y_votos == y_votos[0], 0, 1)
        Xe_votos,ye_votos,Xp_votos,yp_votos=particion_entr_prueba(X_votos,y_votos_lr,test=0.2)
        # No es necesario normalizar este dataset
        Ajusta_parametros(Xe_votos,Xp_votos,ye_votos,yp_votos)
       
        print("##->              Datos de Películas\n")
        # Como los datos de las películas de IMDB ya están vectorizados no es necesario hacer nada
        Ajusta_parametros(X_train_imdb, y_train_imdb, X_test_imdb, y_test_imdb)

testEj5()

# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------

raise



class RL_OvR():

    def __init__(self,rate=0.1,rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.clasificadores = {} 

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        self.clases = np.unique(y)  

        # Entrenamos un clasificador por cada clase
        for clase in self.clases:
            # Creamos una copia de las etiquetas y asignamos todas las instancias que no pertenecen a la clase actual a 0
            y_temp = np.where(y == clase, 1, 0)
            
            # Entrenamiento
            clasificador = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam)
            clasificador.entrena(X, y_temp, n_epochs=n_epochs, salida_epoch=salida_epoch)
            
            # Segun la clase, se le asigna un clasificador u otro
            self.clasificadores[clase] = clasificador

    def clasifica(self, ejemplos):
        # Realizamos la predicción con cada uno de los clasificadores y guardamos las probabilidades de la clase positiva
        probabilidades = np.array([self.clasificadores[clase].clasifica_prob(ejemplos) for clase in self.clases])
        
        # Devolvemos la clase con la probabilidad más alta para cada ejemplo
        return self.clases[np.argmax(probabilidades, axis=0)]

# --------------------------------

# Test
Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

rl_iris_ovr=RL_OvR(rate=0.01,batch_tam=8)

rl_iris_ovr.entrena(Xe_iris,ye_iris)

rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9



            
# --------------------------------







# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles  valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
  
# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])
   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     

    
  

# -------- 

def codifica_one_hot(X):
    matrices_one_hot = []
    
    for i in range(X.shape[1]):
        valores_unicos = np.unique(X[:, i])
        
        # Creamos una matriz de ceros para rellenarla despues
        lista_one_hot = np.zeros((X.shape[0], valores_unicos.shape[0]))
        
        # Se coloca un 1 por cada valor unico en su columna de la lista
        for j, valor_unico in enumerate(valores_unicos):
            lista_one_hot[np.where(X[:, i] == valor_unico), j] = 1
        
        # En la matriz guardamos cada lista rellenada
        matrices_one_hot.append(lista_one_hot)
    
    X_one_hot = np.concatenate(matrices_one_hot, axis=1)
    
    return X_one_hot



# Test
Xc=np.array([     ["a",1,"c","x"],
                  ["b",2,"c","y"],
                  ["c",1,"d","x"],
                  ["a",2,"d","z"],
                  ["c",1,"e","y"],
                  ["c",2,"f","y"]])
   
codifica_one_hot(Xc)
# Salida esperada:
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])




# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------

# Aplicamos codificación one-hot a las características
X_credito_one_hot = codifica_one_hot(X_credito)

# Dividimos los datos en conjuntos de entrenamiento y prueba
Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(X_credito_one_hot, y_credito)


mejor_rendimiento = 0
mejor_rate = 0
mejor_batch_tam = 0
mejor_rate_decay = False


# Búsqueda de parámetros
for rate in [0.1, 0.01, 0.001]:
    for batch_tam in [16, 32, 64, 128]:
            for rate_decay in [True, False]:

                # Crear y entrenar el modelo
                rl_ovr = RL_OvR(rate=rate, batch_tam=batch_tam, rate_decay=rate_decay)
                rl_ovr.entrena(Xe_credito, ye_credito)
                    
                # Evaluar rendimiento
                rend = rendimiento_validacion_cruzada(RL_OvR,{"batch_tam":batch_tam,"rate":rate,"rate_decay":rate_decay}, Xe_credito,ye_credito,n=5)
                print(f"\n rate: {rate}, batch_tam: {batch_tam}, rate_decay: {rate_decay}, rendimiento: {rend} ")    
               
                # Actualizar mejores parámetros si es necesario
                if rend > mejor_rendimiento:
                    mejor_rendimiento = rend
                    mejor_rate = rate
                    mejor_batch_tam = batch_tam

print(f"\n Mejores parámetros encontrados: rate: {mejor_rate}, batch_tam: {mejor_batch_tam}, rendimiento: {mejor_rendimiento}")

# Entrenamos el modelo con los mejores hiperparámetros encontrados
rl_ovr = RL_OvR(rate=mejor_rate, batch_tam=mejor_batch_tam)
rl_ovr.entrena(Xe_credito, ye_credito, salida_epoch=False)






# ---------------------------------------------------------
# 8.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------


# -------------------------------
# TIEMPO DE EJECUCION:
# Aprox 1min 20s
# -------------------------------

# Funciones auxiliares para lectura de ficheros
def read_images(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Transforma el texto a una imagen binaria (28x28)
        images = [np.array([[int(col != ' ') for col in row] for row in lines[i:i+28]]) for i in range(0, len(lines), 28)]
        images = [img.flatten() for img in images]
    return np.array(images)


def read_labels(file_path):
    with open(file_path, 'r') as f:
        labels = np.array([int(label) for label in f.read().split("\n") if label])
    return labels

# Carga de datos
X_train = read_images("datos/digitdata/trainingimages")
y_train = read_labels("datos/digitdata/traininglabels")

X_valid = read_images("datos/digitdata/validationimages")
y_valid = read_labels("datos/digitdata/validationlabels")

X_test = read_images("datos/digitdata/testimages")
y_test = read_labels("datos/digitdata/testlabels")

# Mejores parametros encontrados (ver busqueda de hiperparametros abajo)
rl_digits_ovr = RL_OvR(rate=0.1, batch_tam=32, rate_decay=False)

# Entrenamiento
rl_digits_ovr.entrena(X_train, y_train)

# Evaluacion
rendimiento_entrenamiento = rendimiento(rl_digits_ovr, X_train, y_train)
rendimiento_validacion = rendimiento(rl_digits_ovr, X_valid, y_valid)


# ---------------------------------

# Búsqueda de hiperparámetros

mejor_rendimiento = 0
mejor_rate = 0
mejor_batch_tam = 0
mejor_rate_decay = False


for rate in [0.1, 0.01, 0.001]:
    for batch_tam in [16, 32, 64, 128]:
            for rate_decay in [True, False]:

                rl_digits_ovr = RL_OvR(rate=rate, batch_tam=batch_tam, rate_decay=rate_decay)
                rl_digits_ovr.entrena(X_train, y_train)
                    
                rend = rendimiento(rl_digits_ovr, X_train, y_train)
                print(f"\n rate: {rate}, batch_tam: {batch_tam}, rate_decay: {rate_decay}, rendimiento: {rend} ")    
               
                if rend > mejor_rendimiento:
                    mejor_rendimiento = rend
                    mejor_rate = rate
                    mejor_batch_tam = batch_tam

print(f"\n Mejores parámetros encontrados: rate: {mejor_rate}, batch_tam: {mejor_batch_tam}, rate_decay: {rate_decay},  rendimiento: {mejor_rendimiento}")
# >>> Mejores parámetros encontrados: rate: 0.1, batch_tam: 32, rate_decay: False, rendimiento: 0.988
















# =========================================================================
# EJERCICIO OPCIONAL PARA SUBIR NOTA: 
#    CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA MULTINOMIAL
# =========================================================================


#  Se pide implementar un clasificador para regresión
#  multinomial logística con softmax (VERSIÓN MINIBATCH), descrito en las 
#  diapositivas 55 a 57 del tema de "Complementos de Aprendizaje Automático". 

# class RL_Multinomial():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica_prob(self,ejemplos):

#        ......
 

#     def clasifica(self,ejemplos):

#        ......
   

 
# Los parámetros tiene el mismo significado que en el ejercicio 7 de OvR. 

# En eset caso, tiene sentido definir un clasifica_prob, ya que la función
# softmax nos va a devolver una distribución de probabilidad de pertenecia 
# a las distintas clases. 


# NOTA 1: De nuevo, es muy importante para la eficiencia usar numpy para evitar
#         el uso de bucles for convencionales.  

# NOTA 2: Se recomienda usar la función softmax de scipy.special: 

    # from scipy.special import softmax   
#

    
# --------------------------------------------------------------------

# Ejemplo:

# >>> rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)

# >>> rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# >>> rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

# --------------- 


from scipy.special import softmax

# Se implementa con la misma estructura que el RL tradicional, pero predecimos probabilidades 
# de pertenecer a una de varias clases

class RL_Multinomial:
    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.pesos = None

    def entrena(self, X, y, Xv=None, yv=None, n_epochs=100, salida_epoch=False, early_stopping=True):
        self.clases = np.unique(y)
        num_clases = len(self.clases)
        num_caract = X.shape[1]
        self.pesos = np.random.randn(num_clases, num_caract)
        
        mejor_EC = float('inf')
        cuenta_paciencia = 0

        for epoch in range(n_epochs):
            idx = np.random.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]
            
            for i in range(0, X.shape[0], self.batch_tam):
                X_mini, y_mini = X[i:i+self.batch_tam], y[i:i+self.batch_tam]
                
                y_pred = self.clasifica_prob(X_mini)
                y_mini_one_hot = np.eye(num_clases)[y_mini]
                error = y_mini_one_hot - y_pred
                
                grad = np.dot(error.T, X_mini)
                
                if self.rate_decay:
                    self.rate = self.rate * (1/(1 + epoch))
                    
                self.pesos += self.rate * grad

            if salida_epoch or early_stopping:
                EC_train = self.entropia_cruzada(y, self.clasifica_prob(X))
                if salida_epoch:
                    print(f"Epoch {epoch}, EC: {EC_train}, Rendimiento: {rendimiento(self, X, y)}")
                
                if early_stopping:
                    if Xv is not None and yv is not None:
                        EC_val = self.entropia_cruzada(yv, self.clasifica_prob(Xv))
                        if EC_val < mejor_EC:
                            mejor_EC = EC_val
                            cuenta_paciencia = 0
                        else:
                            cuenta_paciencia += 1
                            if cuenta_paciencia >= self.paciencia:
                                print("Parada temprana")
                                return
                
    # Clasifica_prob ahora devuelve una probabilidad en softmax
    def clasifica_prob(self, X):
        if self.pesos is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado.")
        return softmax(np.dot(X, self.pesos.T), axis=1)
    
    # Devolvemos la clase con probabilidad maxima
    def clasifica(self, X):
        probas = self.clasifica_prob(X)
        return np.argmax(probas, axis=1)
    
    def entropia_cruzada(self, y, y_pred):
        y_one_hot = np.eye(len(self.clases))[y]
        return -np.sum(y_one_hot * np.log(y_pred))


rl_iris_m=RL_Multinomial(rate=0.01,batch_tam=8)
rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50, salida_epoch=True)

rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158

####################################### TESTING ##################################

#testEj1()
#testEj2_1()
#testEj2_2()
#test_Ej3()
#testEj4()
#testEj5()


##################################################################################








