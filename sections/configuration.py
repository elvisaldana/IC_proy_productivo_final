import streamlit as st
import pandas as pd
import os
from supabase_api import fetch_data_from_supabase, insert_data_into_supabase


def configuration_section():
    """
    Opciones Generales de Configuración.
    """
    st.subheader("Opciones Generales")
    st.write("Configuración general del sistema. Aquí puedes realizar ajustes globales.")

def usuarios_section():
    """
    Gestión de Usuarios.
    """
    st.subheader("Gestión de Usuarios")
    st.write("Opciones para gestionar usuarios del sistema.")


def mantenimiento_proveedores_section():
    """
    Sección de mantenimiento de proveedores.
    """
    st.subheader("Mantenimiento de Proveedores")
    st.write("Aquí puedes gestionar los datos de los proveedores.")

    # Botón para consultar proveedores
    if st.button("Consultar Proveedores"):
        try:
            response = fetch_data_from_supabase("proveedor")
            st.success("Datos consultados correctamente:")
            st.dataframe(response)
        except Exception as e:
            st.error(f"Error al consultar la tabla proveedor: {e}")

    # Campos para agregar o actualizar un proveedor
    ruc_proveedor = st.text_input("RUC del Proveedor")
    nombre_proveedor = st.text_input("Nombre del Proveedor")

    if st.button("Agregar o Actualizar Proveedor"):
        if not ruc_proveedor or not nombre_proveedor:
            st.warning("Debe completar todos los campos.")
        else:
            nuevo_proveedor = {
                "ruc_proveedor": ruc_proveedor,
                "nombre_proveedor": nombre_proveedor,
            }
            try:
                st.write("Datos a insertar o actualizar:", nuevo_proveedor)  # Depuración
                response = insert_data_into_supabase("proveedor", nuevo_proveedor)
                st.success("Proveedor agregado o actualizado correctamente.")
                st.write("Respuesta de Supabase:", response.data)  # Mostrar respuesta
            except Exception as e:
                st.error(f"Error al agregar o actualizar el proveedor: {e}")

def mantenimiento_centros_section():
    """
    Mantenimiento de Centro de Coste.
    """
    st.subheader("Mantenimiento de Centro de Coste")
    st.write("Aquí puedes gestionar los datos de los centros de coste.")

    # Implementar lógica similar a Proveedores...

def mantenimiento_productos_section():
    """
    Mantenimiento de Productos.
    """
    st.subheader("Mantenimiento de Productos")
    st.write("Aquí puedes gestionar los datos de los productos.")

    # Implementar lógica similar a Proveedores...

def mantenimiento_estados_section():
    """
    Mantenimiento de Estados.
    """
    st.subheader("Mantenimiento de Estados")
    st.write("Aquí puedes gestionar los datos de los estados.")

    # Implementar lógica similar a Proveedores...

