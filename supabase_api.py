import os
import requests
from dotenv import load_dotenv
from supabase import create_client


# Cargar credenciales desde el archivo .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

HEADERS = {
    "Content-Type": "application/json",
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

def fetch_data_from_supabase(table):
    """
    Recupera datos de una tabla de Supabase.
    :param table: Nombre de la tabla a consultar.
    :return: Lista de diccionarios con los datos de la tabla.
    """
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error al consultar Supabase: {response.status_code} - {response.text}")

def insert_data_into_supabase(table_name, data):
    """
    Inserta o actualiza datos en una tabla de Supabase.
    :param table_name: Nombre de la tabla donde se insertarán los datos.
    :param data: Diccionario con los datos a insertar o actualizar.
    :return: Respuesta de la API en formato JSON.
    """
    try:
        # Realizar la operación de UPSERT
        response = supabase.table(table_name).upsert(data).execute()
        
        # Validar si la respuesta contiene datos
        if hasattr(response, "data") and response.data:  # Verificar atributo "data"
            return response
        else:
            raise ValueError(f"Error en la respuesta de Supabase: {response}")
    except Exception as e:
        raise ValueError(f"Error al insertar datos en Supabase: {e}")










