import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from supabase_api import fetch_data_from_supabase  

def stats_visuals_section():
    st.header("Analisis Descriptivo")

    # 🔹 Cargar datos desde Supabase
    try:
        data = fetch_data_from_supabase("vista_general_ordenes2")
        if not data or len(data) == 0:
            st.warning("⚠ No hay datos disponibles en la tabla vista_general_ordenes2.")
            return

        df = pd.DataFrame(data)
        
        # 🔹 Convertir fechas
        date_cols = ["fecha_creacion_compra", "fecha_aprobacion_compra", "fecha_pedido_compra", "fecha_recepcion"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        # 🔹 Cálculo de métricas clave
        total_productos = df["cantidad"].sum()
        total_ordenes = df["orden_id"].nunique()
        monto_total_general = df["total"].sum()
        monto_por_categoria = df.groupby("categoria_nombre")["total"].sum().reset_index()

        # 🔹 Mostrar métricas en columnas
        st.subheader("Indicadores Claves de Compras")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cantidad Total de Productos", f"{total_productos:,}")
        col2.metric("Cantidad de Órdenes Emitidas", f"{total_ordenes:,}")
        col3.metric("Monto Total de Compras", f"S/. {monto_total_general:,.2f}")

        # 🔹 Monto total por categoría en un DataFrame
        st.subheader("Monto Total por Categoría")
        st.dataframe(monto_por_categoria)
               
        # 🔹 Aplicar Filtros dinámicos en Sidebar
        st.sidebar.header("🔍 Filtros de Búsqueda")
        proveedor = st.sidebar.multiselect("Proveedor", df["proveedor_nombre"].unique())
        categoria = st.sidebar.multiselect("Categoría", df["categoria_nombre"].unique())
        rango_fechas = st.sidebar.date_input("Rango de Fechas", [])

        if proveedor:
            df = df[df["proveedor_nombre"].isin(proveedor)]
        if categoria:
            df = df[df["categoria_nombre"].isin(categoria)]
        if len(rango_fechas) == 2:
            df = df[(df["fecha_creacion_compra"] >= pd.to_datetime(rango_fechas[0])) & 
                    (df["fecha_creacion_compra"] <= pd.to_datetime(rango_fechas[1]))]

        # ⿢ Gráfico de Barras Apiladas - Top 10 Proveedores
        st.subheader("Composición de Compras")

        # Agrupar por proveedor y categoría
        df_grouped = df.groupby(['proveedor_nombre', 'categoria_nombre'])['total'].sum().reset_index()

        # Seleccionar los 10 proveedores con mayor facturación
        top_10_proveedores = df_grouped.groupby('proveedor_nombre')['total'].sum().nlargest(10).index
        df_top_10 = df_grouped[df_grouped['proveedor_nombre'].isin(top_10_proveedores)]

        # Crear el gráfico
        fig = px.bar(df_top_10, 
                    x='proveedor_nombre', 
                    y='total', 
                    color='categoria_nombre', 
                    title="Top 10 Proveedores y Categorías", 
                    text_auto=True)

        # Mostrar gráfico
        st.plotly_chart(fig)

       
        # 📊 Gráfico de Barras: Compras por Categoría
        st.subheader("Compras por Categoría")
        fig_bar = px.bar(df, x="categoria_nombre", y="subtotal", color="proveedor_nombre", barmode="group")
        st.plotly_chart(fig_bar)

        # ⿡⿠ Gráfico de Sankey - Flujo de Presupuesto en Compras (Agrupando Proveedores)
        st.subheader("Flujo de Presupuesto en Compras")

        # 🔹 Obtener los 10 Proveedores principales y agrupar el resto en "Otros"
        top_proveedores = df.groupby('proveedor_nombre')['total'].sum().nlargest(10).index
        df_sankey = df.copy()
        df_sankey['proveedor_nombre'] = df_sankey['proveedor_nombre'].apply(lambda x: x if x in top_proveedores else "Otros Proveedores")

        # 🔹 Agrupar los valores con los proveedores ajustados
        df_sankey = df_sankey.groupby(['centro_costo_area', 'categoria_nombre', 'proveedor_nombre'])['total'].sum().reset_index()

        # 🔹 Generar nodos únicos
        labels = list(pd.concat([df_sankey['centro_costo_area'], df_sankey['categoria_nombre'], df_sankey['proveedor_nombre']]).unique())

        # 🔹 Definir conexiones entre los nodos
        source = df_sankey['centro_costo_area'].apply(lambda x: labels.index(x))
        target = df_sankey['categoria_nombre'].apply(lambda x: labels.index(x))
        target2 = df_sankey['proveedor_nombre'].apply(lambda x: labels.index(x))
        values = df_sankey['total']

        # 🔹 Crear el gráfico de Sankey optimizado
        fig = go.Figure(go.Sankey(
            node=dict(label=labels, pad=15, thickness=20),  # Color más elegante
            link=dict(source=source.tolist() + target.tolist(),
              target=target.tolist() + target2.tolist(),
              value=values.tolist() + values.tolist())
        ))

        st.plotly_chart(fig)

         # ⿧ Gráfico de Área Apilada - Evolución de Compras por Categoría
        st.subheader("Evolución de Compras por Categoría")
        df_time = df.groupby(['fecha_creacion_compra', 'categoria_nombre'])['total'].sum().reset_index()
        fig = px.area(df_time, x='fecha_creacion_compra', y='total', color='categoria_nombre', title="Compras por Categoría a lo Largo del Tiempo")
        st.plotly_chart(fig)

        # ⿣ Gráfico de Sunburst - Top 10 Proveedores con Colores Suaves
        st.subheader("Distribución Jerárquica de Compras")

        # 🔹 Obtener el top 10 proveedores por monto total de compras
        top_10_proveedores = df.groupby('proveedor_nombre')['total'].sum().nlargest(10).index

        # 🔹 Filtrar el dataframe para solo incluir esos proveedores
        df_top = df[df['proveedor_nombre'].isin(top_10_proveedores)]

        # 🔹 Crear el gráfico con una paleta de colores más suave
        fig = px.sunburst(df_top, 
                  path=['categoria_nombre', 'subcategoria_nombre', 'proveedor_nombre'], 
                  values='total', 
                  title="Gráfico Sunburst de Compras (Top 10 Proveedores)",
                  color='proveedor_nombre',  # 🎨 Diferenciar por proveedor
                  color_discrete_sequence=px.colors.sequential.Blues,  # 🌸 Paleta de colores pastel
                  width=900,  
                  height=900)

        st.plotly_chart(fig, use_container_width=True)  # Ajuste automático de ancho

        # 1. Gráfico de dispersión: Relación entre cantidad y total de compra
        st.subheader("Gráfico de Dispersión - Cantidad vs Total de Compra")
        fig_disp = px.scatter(df, x="cantidad", y="total", color="categoria_nombre", title="Relación entre Cantidad y Total de Compra")
        st.plotly_chart(fig_disp)

         # 🔟 Relación entre Cantidad Comprada y Precio del Producto
        st.subheader("Relación entre Cantidad Comprada y Precio del Producto")
        fig = px.scatter(df, x='cantidad', y='producto_precio', color='categoria_nombre', title="Cantidad Comprada vs. Precio del Producto")
        st.plotly_chart(fig)
            
        # 7. Matriz de gráficos de dispersión
        st.subheader("Matriz de Dispersión")
        fig_matrix = px.scatter_matrix(df, dimensions=["cantidad", "total", "producto_precio"], color="categoria_nombre")
        st.plotly_chart(fig_matrix)
                                             
        # ⿩ Top 10 Proveedores con Mayor Facturación
        st.subheader("Top 10 Proveedores con Mayor Facturación")
        top_prov = df.groupby('proveedor_nombre')['total'].sum().nlargest(10).reset_index()
        fig = px.bar(top_prov, x='proveedor_nombre', y='total', title="Top 10 Proveedores con Mayor Facturación", text_auto=True)
        st.plotly_chart(fig)
    
        # ⿡⿢ Top 10 Proveedores por Cantidad de Compras
        st.subheader("Top 10 Proveedores por Cantidad de Compras")
        top_prov_cantidad = df.groupby('proveedor_nombre')['cantidad'].sum().nlargest(10).reset_index()
        fig = px.bar(top_prov_cantidad, x='proveedor_nombre', y='cantidad', title="Top 10 Proveedores por Cantidad de Compras", text_auto=True)
        st.plotly_chart(fig)
              
        # ⿡⿤ Top 10 de Productos con Mayor Frecuencia de Compra
        st.subheader("Top 10 Productos con Mayor Frecuencia de Compra")
        top_productos = df.groupby('producto_descripcion')['cantidad'].sum().nlargest(10).reset_index()
        fig = px.bar(top_productos, x='producto_descripcion', y='cantidad', title="Top 10 Productos con Mayor Frecuencia de Compra", text_auto=True)
        st.plotly_chart(fig)
   
        st.write("Fin")
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")