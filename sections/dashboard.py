import streamlit as st
import pandas as pd
import plotly.express as px
from supabase_api import fetch_data_from_supabase

def dashboard_section():
    st.title("Dashboard Interactivo")
    st.subheader("Visión general de las Compras por Categoría")

    # Cargar datos desde Supabase
    try:
        table_name = "vista_categorias_compras"
        data = fetch_data_from_supabase(table_name)
        df = pd.DataFrame(data)

        st.write("Datos cargados (vista previa):")
        st.dataframe(df.head())

        # Filtros dinámicos
        st.sidebar.subheader("Filtros")
        
        # Filtro de Categoría
        categorias = st.sidebar.multiselect(
            "Categorías",
            options=df["categoria"].unique(),
            default=df["categoria"].unique()
        )
        
        # Filtro de Subcategoría
        subcategorias = st.sidebar.multiselect(
            "Subcategorías",
            options=df["subcategoria"].unique(),
            default=df["subcategoria"].unique()
        )
        
        # Aplicar filtros
        if categorias:
            df = df[df["categoria"].isin(categorias)]
        
        if subcategorias:
            df = df[df["subcategoria"].isin(subcategorias)]

        # Métricas clave
        st.subheader("Métricas Clave")
        st.metric("Total de Compras", len(df))
        st.metric("Categorías Distintas", df["categoria"].nunique())
        st.metric("Subcategorías Distintas", df["subcategoria"].nunique())

        # Gráficos
        st.subheader("Visualizaciones")
        
        # Gráfico de barras: Distribución de compras por categoría
        if "categoria" in df.columns:
            st.plotly_chart(px.bar(df, x="categoria", title="Distribución de Compras por Categoría", color="categoria"))
        
        # Gráfico circular: Proporción de compras por subcategoría
        if "subcategoria" in df.columns:
            st.plotly_chart(px.pie(df, names="subcategoria", title="Proporción de Compras por Subcategoría"))

    except Exception as e:
        st.error(f"Error al cargar datos para el Dashboard: {e}")
