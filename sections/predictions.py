import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from supabase_api import fetch_data_from_supabase
from sklearn.preprocessing import LabelEncoder, StandardScaler

def predictions_section():
    st.header("📈 Predicciones para la Gerencia de Presupuestos")

    # 📌 **Cargar datos desde Supabase**
    table_name = "vista_train_model"
    try:
        data = fetch_data_from_supabase(table_name)
        df = pd.DataFrame(data)
        st.write("📥 **Vista previa de los datos cargados:**", df.head())  
    except Exception as e:
        st.error(f"❌ Error al cargar datos desde Supabase: {e}")
        return

    if df.empty:
        st.warning("⚠️ No hay datos disponibles para hacer predicciones.")
        return

    # ✅ **Mostrar columnas disponibles**
    st.write("📋 **Columnas disponibles en la vista:**", df.columns.tolist())

    # ✅ **Seleccionar el modelo a usar**
    modelo_tipo = st.selectbox("📌 Seleccione el modelo de predicción:", 
                               ["🛒 Clasificación del Tipo de Compra", "⏳ Predicción de Tiempos de Entrega"])

    # ✅ **Cargar el modelo entrenado correspondiente**
    modelo_filename = f"modelo_{modelo_tipo.replace(' ', '_')}.pkl"

    try:
        modelo = joblib.load(modelo_filename)
        st.success(f"✅ Modelo `{modelo_tipo}` cargado correctamente.")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo `{modelo_tipo}`: {e}")
        return

    # **Definir características para cada modelo**
    if modelo_tipo == "🛒 Clasificación del Tipo de Compra":
        st.subheader("🛒 Clasificación de Tipo de Compra")
        features = ["proveedor_id", "producto_id", "cantidad", "impuestos"]
    elif modelo_tipo == "⏳ Predicción de Tiempos de Entrega":
        st.subheader("⏳ Predicción de Tiempos de Entrega")
        features = ["proveedor_id", "producto_id", "cantidad", "impuestos"]

    # ✅ **Verificar que las características están en los datos**
    if not all(col in df.columns for col in features):
        st.error("❌ Error: Algunas columnas necesarias para la predicción no están en los datos.")
        return

    # ✅ **Preprocesamiento de los datos**
    # 🔹 Convertir variables categóricas
    categorical_cols = ["proveedor_id", "producto_id"]
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # 🔹 Escalar variables numéricas
    scaler = StandardScaler()
    numeric_cols = ["cantidad", "impuestos"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ✅ **Realizar predicciones**
    X = df[features]
    predictions = modelo.predict(X)

    # ✅ **Agregar predicciones al DataFrame**
    df["prediccion"] = predictions

    # 📌 **Mostrar resultados**
    st.write("📊 **Resultados de la Predicción:**", df[["proveedor_id", "producto_id", "cantidad", "prediccion"]])

    # 📈 **Visualización de resultados**
    if modelo_tipo == "🛒 Clasificación del Tipo de Compra":
        fig = px.bar(df, x="proveedor_id", y="prediccion", title="Clasificación del Tipo de Compra", labels={"prediccion": "Categoría Predicha"})
    else:
        fig = px.scatter(df, x="cantidad", y="prediccion", title="Predicción de Tiempos de Entrega", labels={"prediccion": "Días Estimados"})
    
    st.plotly_chart(fig)

    # 📘 **Guía Rápida para el Usuario**
    st.info(f"""
    📘 **Guía Rápida**  
    - 🛒 **Clasificación del Tipo de Compra**: Ayuda a predecir si una compra es de tipo *producto* o *servicio* en función de los datos históricos.  
    - ⏳ **Predicción de Tiempos de Entrega**: Permite estimar cuántos días tomará recibir un pedido basándose en el proveedor y el tipo de producto.  
    - 🔍 **Cómo Interpretar los Resultados**:  
      - En **Clasificación**, cada número representa una categoría predicha.  
      - En **Tiempos de Entrega**, valores altos indican mayor demora.  
    """)

    # 📥 **Botón para exportar resultados**
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Resultados en CSV", csv, f"resultados_{modelo_tipo}.csv", "text/csv", key="download-csv")
