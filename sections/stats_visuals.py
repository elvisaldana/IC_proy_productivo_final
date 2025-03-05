import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from supabase_api import fetch_data_from_supabase  

def stats_visuals_section():
    st.header("Analisis Descriptivo")

    # 🔹 Cargar datos desde Supabase
    try:
        data = fetch_data_from_supabase("vista_general_ordenes")
        if not data or len(data) == 0:
            st.warning("⚠ No hay datos disponibles en la tabla vista_general_ordenes.")
            return

        df = pd.DataFrame(data)

        # 🔹 Convertir fechas
        date_cols = ["fecha_creacion_compra", "fecha_aprobacion_compra", "fecha_pedido_compra", "fecha_recepcion"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

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

        # ✅ Mostrar vista previa de los datos filtrados
        st.subheader("Vista previa de los datos filtrados")
        st.dataframe(df.head())

        # 2️⃣ Gráfico de Barras Apiladas
        st.subheader("Composición de Compras por Proveedor y Categoría")
        df_grouped = df.groupby(['proveedor_nombre', 'categoria_nombre'])['total'].sum().reset_index()
        fig = px.bar(df_grouped, x='proveedor_nombre', y='total', color='categoria_nombre', title="Barras Apiladas: Proveedores y Categorías", text_auto=True)
        st.plotly_chart(fig)

        # 📊 Gráfico de Barras: Compras por Categoría
        st.subheader("Compras por Categoría")
        fig_bar = px.bar(df, x="categoria_nombre", y="subtotal", color="proveedor_nombre", barmode="group")
        st.plotly_chart(fig_bar)

        # 3️⃣ Gráfico de Sunburst
        st.subheader("Distribución Jerárquica de Compras")
        fig = px.sunburst(df, path=['categoria_nombre', 'subcategoria_nombre', 'proveedor_nombre'], values='total', title="Gráfico Sunburst de Compras")
        st.plotly_chart(fig)

        # 📦 Diagrama de Caja: Comparación de Precios
        st.subheader("Análisis de Precios por Categoría")
        fig_caja = px.box(df, x="categoria_nombre", y="producto_precio", color="categoria_nombre")
        st.plotly_chart(fig_caja)

        # 1. Gráfico de dispersión: Relación entre cantidad y total de compra
        st.subheader("Gráfico de Dispersión - Cantidad vs Total de Compra")
        fig_disp = px.scatter(df, x="cantidad", y="total", color="categoria_nombre", title="Relación entre Cantidad y Total de Compra")
        st.plotly_chart(fig_disp)
        
        # 2. Histograma de distribución de compras
        st.subheader("Histograma de Compras por Categoría")
        fig_hist = px.histogram(df, x="categoria_nombre", y="cantidad", color="categoria_nombre", title="Distribución de Compras por Categoría")
        st.plotly_chart(fig_hist)
        
        # 3. Mapa de calor de correlaciones
        st.subheader("Mapa de Calor - Correlaciones")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # 5. Diagrama de caja - Análisis de costos
        st.subheader("Diagrama de Caja - Distribución de Costos")
        fig_box = px.box(df, x="categoria_nombre", y="total", color="categoria_nombre", title="Variabilidad en Costos por Categoría")
        st.plotly_chart(fig_box)
        
        # 6. Diagrama de Pareto - Identificación de los proveedores más relevantes
        st.subheader("Diagrama de Pareto - Proveedores")
        proveedor_costos = df.groupby("proveedor_nombre")["total"].sum().reset_index().sort_values(by="total", ascending=False)
        proveedor_costos["porcentaje_acumulado"] = proveedor_costos["total"].cumsum() / proveedor_costos["total"].sum()
        fig_pareto = px.bar(proveedor_costos, x="proveedor_nombre", y="total", title="Distribución de Costos por Proveedor")
        fig_pareto.add_scatter(x=proveedor_costos["proveedor_nombre"], y=proveedor_costos["porcentaje_acumulado"] * proveedor_costos["total"].max(), mode="lines", name="% Acumulado")
        st.plotly_chart(fig_pareto)
        
        # 7. Matriz de gráficos de dispersión
        st.subheader("Matriz de Dispersión")
        fig_matrix = px.scatter_matrix(df, dimensions=["cantidad", "total", "producto_precio"], color="categoria_nombre")
        st.plotly_chart(fig_matrix)
        
           # 7️⃣ Gráfico de Área Apilada - Evolución de Compras por Categoría
        st.subheader("Evolución de Compras por Categoría")
        df_time = df.groupby(['fecha_creacion_compra', 'categoria_nombre'])['total'].sum().reset_index()
        fig = px.area(df_time, x='fecha_creacion_compra', y='total', color='categoria_nombre', title="Compras por Categoría a lo Largo del Tiempo")
        st.plotly_chart(fig)
        
        # 1️⃣0️⃣ Gráfico de Sankey - Flujo de Presupuesto en Compras
        st.subheader("Flujo de Presupuesto en Compras")
        df_sankey = df.groupby(['centro_costo_area', 'categoria_nombre', 'proveedor_nombre'])['total'].sum().reset_index()
        labels = list(pd.concat([df_sankey['centro_costo_area'], df_sankey['categoria_nombre'], df_sankey['proveedor_nombre']]).unique())
        source = df_sankey['centro_costo_area'].apply(lambda x: labels.index(x))
        target = df_sankey['categoria_nombre'].apply(lambda x: labels.index(x))
        target2 = df_sankey['proveedor_nombre'].apply(lambda x: labels.index(x))
        values = df_sankey['total']
    
        fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=20),
        link=dict(source=source.tolist() + target.tolist(), target=target.tolist() + target2.tolist(), value=values.tolist() + values.tolist())
        ))
        st.plotly_chart(fig)
                   
        # 9️⃣ Top 10 Proveedores con Mayor Facturación
        st.subheader("Top 10 Proveedores con Mayor Facturación")
        top_prov = df.groupby('proveedor_nombre')['total'].sum().nlargest(10).reset_index()
        fig = px.bar(top_prov, x='proveedor_nombre', y='total', title="Top 10 Proveedores con Mayor Facturación", text_auto=True)
        st.plotly_chart(fig)
    
        # 🔟 Relación entre Cantidad Comprada y Precio del Producto
        st.subheader("Relación entre Cantidad Comprada y Precio del Producto")
        fig = px.scatter(df, x='cantidad', y='producto_precio', color='categoria_nombre', title="Cantidad Comprada vs. Precio del Producto")
        st.plotly_chart(fig)
    
        # 1️⃣2️⃣ Top 10 Proveedores por Cantidad de Compras
        st.subheader("Top 10 Proveedores por Cantidad de Compras")
        top_prov_cantidad = df.groupby('proveedor_nombre')['cantidad'].sum().nlargest(10).reset_index()
        fig = px.bar(top_prov_cantidad, x='proveedor_nombre', y='cantidad', title="Top 10 Proveedores por Cantidad de Compras", text_auto=True)
        st.plotly_chart(fig)
    
        # 1️⃣3️⃣ Monto Total Comprado por Área
        st.subheader("Monto Total Comprado por Área")
        df_area = df.groupby('centro_costo_area')['total'].sum().reset_index()
        fig = px.bar(df_area, x='centro_costo_area', y='total', title="Monto Total Comprado por Área", text_auto=True)
        st.plotly_chart(fig)
    
        # 1️⃣4️⃣ Top 10 de Productos con Mayor Frecuencia de Compra
        st.subheader("Top 10 Productos con Mayor Frecuencia de Compra")
        top_productos = df.groupby('producto_descripcion')['cantidad'].sum().nlargest(10).reset_index()
        fig = px.bar(top_productos, x='producto_descripcion', y='cantidad', title="Top 10 Productos con Mayor Frecuencia de Compra", text_auto=True)
        st.plotly_chart(fig)
    
        st.write("Fin")
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")