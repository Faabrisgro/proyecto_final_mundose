import matplotlib.pyplot as plt
import seaborn as sns

def vars_categoricas_graf(dataframe, x, y, use_mean=False, title=None):
    '''
    Función que permite graficar variables categóricas con gráficos de barras.
    Existe la condición de que al ingresar churn dos veces se crea un gráfico diferente, en el resto de los otros casos se procede normalmente.
    '''
    if x == y:
        churn_counts = dataframe[x].value_counts().reset_index()
        churn_counts = churn_counts.replace({0: 'No', 1: 'Sí'})
        sns.barplot(data=churn_counts, x='churn', y='count', hue='churn', legend=False)
        plt.title('Cantidad de clientes que se dieron de baja')
        plt.ylabel('Cantidad de bajas de clientes')
        plt.xlabel('¿El cliente se dio de baja?')
        plt.show()
        print(churn_counts)
    else:
        agg_func = 'mean' if use_mean else 'sum'
        title_prefix = f'Promedio de {y} por ' if use_mean else f'Cantidad de {y} por '
        grouped_df = dataframe.groupby(x)[y].agg(agg_func).reset_index()

        plt.subplots(figsize=(8, 6))
        sns.barplot(data=grouped_df, x=x, y=y, hue=x, legend=False)
        plt.xticks(rotation=90)
        plt.xlabel(f'{title_prefix}{x}')
        plt.ylabel(f'Cantidad de {y}')
        plt.title(title or f'{title_prefix}{x}')
        plt.show()


def multi_vars_graf(dataframe, var1, var2 ,y): #Función que permite graficar variables categóricas y ver la cantidad de renuncias por categoría

    ''' Función que permite graficar multiples variables categóricas con gráficos de barras.'''

    grouped_df = dataframe.groupby([var1, var2])[y].sum().reset_index()
    plt.subplots(figsize=(8,6))
    sns.barplot(data=grouped_df, x=var1, y=y, hue=var2)
    plt.xticks(rotation=90)
    plt.legend(title='Compañías')
    plt.xlabel('Estados')
    plt.ylabel('Cantidad de bajas')
    plt.title(f'Cantidad de bajas por {var1} y {var2}')
    plt.show()
    print('Top 5 variables con mas churn', '\n', '\n', grouped_df.sort_values(by=y, ascending=False).head(5))
    print(dataframe.groupby([var1, var2])[y].sum().idxmax())


def plot_max_churn(df, x_col, y_col, hue_col, title, palette=None, plot_type='bar'):
    # Verificar si x_col y hue_col son iguales
    if x_col == hue_col:
        # Solo agrupar por x_col
        max_churn_by_state = df.groupby(x_col)[y_col].sum().reset_index()
    else:
        # Agrupar por ambas columnas
        max_churn_by_state = df.groupby([x_col, hue_col])[y_col].sum().reset_index()
        max_churn_by_state = max_churn_by_state.loc[max_churn_by_state.groupby(x_col)[y_col].idxmax()]

    # Configurar subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Elegir el tipo de gráfico
    if plot_type == 'bar':
        sns.barplot(data=max_churn_by_state, x=x_col, y=y_col, hue=hue_col, ax=ax, palette=palette)
    elif plot_type == 'barstacked':
        sns.barplot(data=max_churn_by_state, x=x_col, y=y_col, hue=hue_col, ax=ax, palette=palette, dodge=False)
    else:
        raise ValueError("Tipo de gráfico no válido. Use 'bar' o 'barstacked'.")

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(f'Máxima cantidad de {y_col}')
    ax.set_xticks(range(len(max_churn_by_state[x_col])))
    ax.set_xticklabels(max_churn_by_state[x_col], rotation=90)

    # Agregar leyenda si se utiliza el parámetro hue
    if hue_col and x_col != hue_col:
        ax.legend(title=hue_col)

    plt.show()

# Ejemplo de uso:
# plot_max_churn(df, 'state', 'churn', 'telecom_partner', 'Máxima cantidad de churn por estado y compañía de telecomunicaciones', palette='viridis')



def distribution_plot(df, columns, nrows=1, ncols=1):
    plt.figure(figsize=(15, 12))
    
    if nrows * ncols < len(columns):
        raise ValueError("El número de filas y columnas no es suficiente para mostrar todos los gráficos.")
    
    for i, column in enumerate(columns, start=1):
        plt.subplot(nrows, ncols, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribución de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.show()