import streamlit as st
import pandas as pd
from supabase_api import insert_data_into_supabase, fetch_data_from_supabase
from supabase import create_client
from dotenv import load_dotenv
import os

# Configurar conexión a Supabase
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def mapear_campos(df, supabase):
    """
    Mapea los valores de campos relacionados (`codigo_producto`, `ruc_proveedor`, `centro_de_coste`)
    a sus respectivas claves foráneas en la tabla `ordencompra`.
    """
    try:
        # Recuperar datos de las tablas relacionadas
        productos = supabase.table("producto").select("id, codigo_producto").execute().data
        proveedores = supabase.table("proveedor").select("id, ruc_proveedor").execute().data
        centros = supabase.table("centrodecoste").select("id, centro_de_coste").execute().data

        # Crear diccionarios de mapeo
        producto_map = {p["codigo_producto"]: p["id"] for p in productos}
        proveedor_map = {p["ruc_proveedor"]: p["id"] for p in proveedores}
        centro_map = {c["centro_de_coste"]: c["id"] for c in centros}

        # Mapear los campos en el DataFrame
        df["producto_id"] = df["codigo_producto"].map(producto_map)
        df["proveedor_id"] = df["ruc_proveedor"].map(proveedor_map)
        df["centrodecoste_id"] = df["centro_de_coste"].map(centro_map)

        # Verificar si hay valores no mapeados
        errores = []
        if df["producto_id"].isnull().any():
            errores.append("Algunos `codigo_producto` no fueron encontrados en la tabla `producto`.")
        if df["proveedor_id"].isnull().any():
            errores.append("Algunos `ruc_proveedor` no fueron encontrados en la tabla `proveedor`.")
        if df["centrodecoste_id"].isnull().any():
            errores.append("Algunos `centro_de_coste` no fueron encontrados en la tabla `centrodecoste`.")

        if errores:
            st.warning("\n".join(errores))
        else:
            st.success("Todos los campos han sido mapeados correctamente.")
        
        return df
    except Exception as e:
        st.error(f"Error durante el mapeo de campos: {e}")
        return df

def subir_y_mapear_datos():
    """
    Subida y preparación de datos en la tabla ordencompra.
    """
    st.header("Subida y Preparación de Datos")

    # Subida del archivo
    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel para cargar en la tabla", type=["csv", "xlsx"])

    if uploaded_file:
        # Leer el archivo
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Datos cargados (vista previa):")
        st.dataframe(df.head())

        # Validación de columnas requeridas
        required_columns = {
            "codigo_de_compra", "usuario_comprador", "tipo_compra", "cantidad",
            "impuestos", "estado", "fecha_pedido_compra", "fecha_creacion_compra",
            "fecha_aprobacion_compra", "fecha_recepcion", "producto_id",
            "proveedor_id", "centrodecoste_id"
        }

        if not required_columns.issubset(df.columns):
            st.error(f"El archivo no contiene las columnas necesarias: {', '.join(required_columns)}")
        else:
            st.success("La plantilla contiene las columnas necesarias.")
            
            # Validación de tipos de datos antes de la inserción
            try:
                # Validar y convertir tipos de datos explícitamente
                df["cantidad"] = pd.to_numeric(df["cantidad"], errors="raise")
                df["impuestos"] = pd.to_numeric(df["impuestos"], errors="raise")
                df["estado"] = pd.to_numeric(df["estado"], errors="raise")
                df["fecha_pedido_compra"] = pd.to_datetime(df["fecha_pedido_compra"], errors="raise")
                df["fecha_creacion_compra"] = pd.to_datetime(df["fecha_creacion_compra"], errors="raise")
                df["fecha_aprobacion_compra"] = pd.to_datetime(df["fecha_aprobacion_compra"], errors="raise")
                df["fecha_recepcion"] = pd.to_datetime(df["fecha_recepcion"], errors="raise")

                st.success("Los datos cumplen con los tipos requeridos.")

            except Exception as e:
                st.error(f"Error en la validación de tipos de datos: {e}")
                return

            # Mostrar datos validados
            st.subheader("Datos validados listos para insertar en la tabla `ordencompra`:")
            st.dataframe(df)

            # Botón para insertar en la tabla `ordencompra`
        if st.button("Insertar Datos en `ordencompra`"):
            try:
                # Forzar conversión de las columnas clave a enteros
                df["producto_id"] = df["producto_id"].astype(float).astype(int)
                df["proveedor_id"] = df["proveedor_id"].astype(float).astype(int)
                df["centrodecoste_id"] = df["centrodecoste_id"].astype(float).astype(int)

                # Imprimir los valores después de la conversión en formato horizontal
                st.write("Valores convertidos de producto_id:", ", ".join(map(str, df["producto_id"].unique())))
                st.write("Valores convertidos de proveedor_id:", ", ".join(map(str, df["proveedor_id"].unique())))
                st.write("Valores convertidos de centrodecoste_id:", ", ".join(map(str, df["centrodecoste_id"].unique())))


                # Insertar datos fila por fila
                for _, row in df.iterrows():
                    data_to_insert = {
                        "codigo_de_compra": row["codigo_de_compra"],
                        "usuario_comprador": row["usuario_comprador"],
                        "tipo_compra": row["tipo_compra"],
                        "cantidad": row["cantidad"],
                        "impuestos": row["impuestos"],
                        "estado": row["estado"],
                        "fecha_pedido_compra": str(row["fecha_pedido_compra"]),
                        "fecha_creacion_compra": str(row["fecha_creacion_compra"]),
                        "fecha_aprobacion_compra": str(row["fecha_aprobacion_compra"]),
                        "fecha_recepcion": str(row["fecha_recepcion"]),
                        "producto_id": row["producto_id"],
                        "proveedor_id": row["proveedor_id"],
                        "centrodecoste_id": row["centrodecoste_id"],
                    }
                    insert_data_into_supabase("ordencompra", data_to_insert)

                st.success("Los datos fueron insertados correctamente en la tabla `ordencompra`.")
            except Exception as e:
                st.error(f"Error al insertar datos: {e}")

def preparar_datos():
    """
    Preparación de datos: detecta inconsistencias y recomienda correcciones sin guardar los datos.
    """
    st.header("Preparación de Datos")

    # Botón para mostrar la preparación de datos
    if not st.button("Mostrar Preparación de Datos"):
        st.info("Presione el botón para analizar la preparación de datos.")
        return

    # Consultar los datos subidos en la tabla `ordencompra`
    try:
        data = fetch_data_from_supabase("ordencompra")
        if not data:
            st.warning("No hay datos disponibles en la tabla `ordencompra` para preparar.")
            return

        df = pd.DataFrame(data)
        st.write("Datos cargados desde Supabase:")
        st.dataframe(df)

        # Resumen de problemas detectados
        st.subheader("Resumen de Problemas Detectados")

        # Verificar valores nulos
        nulos = df.isnull().sum()
        st.write("Valores nulos por columna:")
        st.dataframe(nulos[nulos > 0])  # Mostrar solo columnas con valores nulos

        # Verificar tipos de datos y formato
        tipos_esperados = {
            "codigo_de_compra": "string",
            "usuario_comprador": "string",
            "tipo_compra": "string",
            "cantidad": "float",
            "impuestos": "float",
            "estado": "int",
            "fecha_pedido_compra": "timestamp",
            "fecha_creacion_compra": "timestamp",
            "fecha_aprobacion_compra": "timestamp",
            "fecha_recepcion": "timestamp",
            "producto_id": "int",
            "proveedor_id": "int",
            "centrodecoste_id": "int"
        }

        errores_tipos = {}
        for col, tipo in tipos_esperados.items():
            if col in df.columns:
                if tipo == "timestamp":
                    # Validar si las fechas están en el formato correcto
                    try:
                        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                        if df[col].isnull().any():
                            errores_tipos[col] = "Algunos valores no son fechas válidas o no cumplen con el formato 'DD/MM/YYYY HH:MM:SS'."
                    except Exception:
                        errores_tipos[col] = "No es un tipo timestamp o no tiene el formato correcto."
                elif tipo == "float":
                    if not pd.api.types.is_float_dtype(df[col]):
                        errores_tipos[col] = "Se esperaba float, pero no tiene el formato correcto."
                elif tipo == "int":
                    if not pd.api.types.is_integer_dtype(df[col]):
                        errores_tipos[col] = "Se esperaba int, pero no tiene el formato correcto."
                elif tipo == "string":
                    if not pd.api.types.is_string_dtype(df[col]):
                        errores_tipos[col] = "Se esperaba string, pero no tiene el formato correcto."

        if errores_tipos:
            st.write("Errores detectados en los tipos de datos:")
            for col, error in errores_tipos.items():
                st.write(f"- {col}: {error}")

        # Verificar valores fuera de rango
        if "cantidad" in df.columns:
            fuera_rango = df[df["cantidad"] < 0]
            if not fuera_rango.empty:
                st.write("Valores fuera de rango detectados en `cantidad` (valores negativos):")
                st.dataframe(fuera_rango)

        # Mostrar recomendaciones
        st.subheader("Recomendaciones")
        if nulos.any() or errores_tipos or not fuera_rango.empty:
            st.warning("Se detectaron problemas en los datos. Corrija los errores en el archivo de importación y vuelva a cargar los datos.")
        else:
            st.success("Los datos no presentan problemas importantes.")

        # Mostrar un cuadro detallado con los tipos de datos esperados
        st.subheader("Tipos de Datos Esperados para `ordencompra`")
        tipos_df = pd.DataFrame.from_dict(tipos_esperados, orient="index", columns=["Tipo Esperado"])
        st.dataframe(tipos_df)

    except Exception as e:
        st.error(f"Error al cargar datos desde Supabase: {e}")
