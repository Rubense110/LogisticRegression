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