import PyPDF2
import pandas as pd
from collections import Counter

def contar_frecuencia_palabras_pdf(ruta_pdf):
    try:
        # Abrir el archivo PDF en modo lectura binaria
        with open(ruta_pdf, 'rb') as archivo:
            # Crear un lector de PDF
            lector = PyPDF2.PdfReader(archivo)
            num_paginas = len(lector.pages)

            # Inicializar una variable para almacenar todo el texto
            texto_completo = ""

            # Extraer el texto de cada página
            for pagina in range(num_paginas):
                pagina_pdf = lector.pages[pagina]
                texto_completo += pagina_pdf.extract_text()

            # Contar la frecuencia de las palabras en el texto completo
            palabras = texto_completo.split()
            frecuencia_palabras = Counter(palabras)

            return frecuencia_palabras
    except Exception as e:
        print(f"Error al procesar el PDF: {e}")
        return None

def guardar_frecuencia_palabras_csv(frecuencia_palabras, ruta_salida):
    # Crear un DataFrame a partir del diccionario de frecuencias
    df = pd.DataFrame(frecuencia_palabras.items(), columns=['Palabra', 'Frecuencia'])
    
    # Guardar el DataFrame en un archivo CSV
    df.to_csv(ruta_salida, index=False)
    print(f"Frecuencia de palabras guardada en {ruta_salida}")

def guardar_frecuencia_palabras_excel(frecuencia_palabras, ruta_salida):
    # Crear un DataFrame a partir del diccionario de frecuencias
    df = pd.DataFrame(frecuencia_palabras.items(), columns=['Palabra', 'Frecuencia'])
    
    # Guardar el DataFrame en un archivo Excel
    df.to_excel(ruta_salida, index=False)
    print(f"Frecuencia de palabras guardada en {ruta_salida}")

# Ruta del archivo PDF
ruta_pdf = 'El_nino.pdf'

# Contar la frecuencia de las palabras en el PDF
frecuencia_palabras = contar_frecuencia_palabras_pdf(ruta_pdf)

if frecuencia_palabras is not None:
    # Guardar las frecuencias en un archivo CSV
    ruta_salida_csv = 'frecuencia_palabras.csv'
    guardar_frecuencia_palabras_csv(frecuencia_palabras, ruta_salida_csv)

    # Guardar las frecuencias en un archivo Excel
    ruta_salida_excel = 'frecuencia_palabras.xlsx'
    guardar_frecuencia_palabras_excel(frecuencia_palabras, ruta_salida_excel)
