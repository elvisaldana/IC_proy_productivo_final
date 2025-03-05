import streamlit as st
from supabase_api import supabase  # Importar supabase desde su archivo centralizado
from sections.data_upload import subir_y_mapear_datos
from sections.data_upload import preparar_datos
from sections.stats_visuals import stats_visuals_section
from sections.train_model import train_model_section
from sections.predictions import predictions_section
from sections.dashboard import dashboard_section
from sections.clustering import clustering_proveedores
from sections.configuration import (
    configuration_section,
    usuarios_section,
    mantenimiento_proveedores_section,
    mantenimiento_centros_section,
    mantenimiento_productos_section,
    mantenimiento_estados_section,
)

st.set_page_config(page_title="Gestión de OrdenCompra", layout="wide")

# Menú lateral con el orden actualizado
menu = st.sidebar.radio(
    "Seleccione una sección:",
    [
        "Subida y Actualizacion de Datos",
        "Estadísticas y Visualización",
        "Clustering",
        "Entrenamiento de Modelos",
        "Predicciones",
        "Dashboard",
        "Configuración",
    ]
)

# Navegación entre secciones según el menú
if menu == "Subida y Actualizacion de Datos":
    subir_y_mapear_datos()
    preparar_datos()
elif menu == "Estadísticas y Visualización":
    stats_visuals_section()
elif menu == "Clustering":
    clustering_proveedores()
elif menu == "Entrenamiento de Modelos":
    train_model_section()
elif menu == "Predicciones":
    predictions_section()
elif menu == "Dashboard":
    dashboard_section()
elif menu == "Configuración":
    st.header("Configuración")
    sub_menu = st.radio(
        "Seleccione una subsección:",
        ["Opciones Generales", "Usuarios", "Mantenimiento"]
    )

    if sub_menu == "Opciones Generales":
        configuration_section()
    elif sub_menu == "Usuarios":
        usuarios_section()
    elif sub_menu == "Mantenimiento":
        mantenimiento_sub_menu = st.radio(
            "Seleccione un área de mantenimiento:",
            ["Proveedores", "Centro de coste", "Productos", "Estados"]
        )

        if mantenimiento_sub_menu == "Proveedores":
            mantenimiento_proveedores_section()  
        elif mantenimiento_sub_menu == "Centro de coste":
            mantenimiento_centros_section()
        elif mantenimiento_sub_menu == "Productos":
            mantenimiento_productos_section()
        elif mantenimiento_sub_menu == "Estados":
            mantenimiento_estados_section()








