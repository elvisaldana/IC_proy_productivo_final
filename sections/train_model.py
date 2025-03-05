import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from supabase_api import fetch_data_from_supabase

def train_model_section():
    st.header("📊 Entrenamiento de Modelos")

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
        st.warning("⚠️ No hay datos disponibles para entrenar.")
        return

    # ✅ **Mostrar las columnas disponibles para depuración**
    st.write("📋 **Columnas disponibles en la vista:**", df.columns.tolist())

    # 📌 **Preprocesamiento de Datos**
    st.subheader("🔍 Preprocesamiento de Datos")

    # ✅ **Convertir variables categóricas a números**
    categorical_cols = ["tipo_compra", "proveedor_id", "producto_id"]
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # ✅ **Escalar variables numéricas para mejorar el entrenamiento**
    scaler = StandardScaler()
    numeric_cols = ["cantidad", "impuestos", "tiempo_entrega"]  # 🔹 Se usa tiempo_entrega en lugar de calcular días
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ✅ **Seleccionar el modelo a entrenar**
    modelo_tipo = st.selectbox("📌 Seleccione el modelo a entrenar:", 
                               ["🛒 Clasificación del Tipo de Compra", "⏳ Predicción de Tiempos de Entrega"])

    # **Inicializar variables vacías para evitar errores**
    features, target = [], ""

    if modelo_tipo == "🛒 Clasificación del Tipo de Compra":
        st.subheader("🛒 Entrenando Modelo de Clasificación de Compras")
        features = ["proveedor_id", "producto_id", "cantidad", "impuestos"]
        target = "tipo_compra"
        modelo = RandomForestClassifier()

    elif modelo_tipo == "⏳ Predicción de Tiempos de Entrega":
        st.subheader("⏳ Entrenando Modelo de Predicción de Tiempos de Entrega")
        features = ["proveedor_id", "producto_id", "cantidad", "impuestos"]
        target = "tiempo_entrega"
        modelo = RandomForestRegressor()

    # ✅ **Validar que features y target están correctamente definidos**
    if not features or target not in df.columns:
        st.warning("⚠️ No se han seleccionado características o variable objetivo para el modelo.")
        return

    # ✅ **Validar que las columnas existen en la vista**
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        st.error(f"❌ Columnas faltantes en los datos: {', '.join(missing_features)}")
        return

    # ✅ **Preparar datos para el entrenamiento**
    X = df[features]
    y = df[target]

    # ✅ **Convertir variables de clasificación a números**
    if modelo_tipo == "🛒 Clasificación del Tipo de Compra":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ✅ **Dividir los datos en entrenamiento y prueba**
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ **Entrenar modelo**
    modelo.fit(X_train, y_train)

    # 📊 **Evaluar rendimiento**
    y_pred = modelo.predict(X_test)
    if modelo_tipo == "⏳ Predicción de Tiempos de Entrega":
        error = mean_absolute_error(y_test, y_pred)
        st.write(f"📏 **Error Medio Absoluto (MAE):** {error:.2f}")
    else:
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"✅ **Precisión del Modelo:** {accuracy:.2%}")

    # 📥 **Guardar modelo entrenado**
    modelo_filename = f"modelo_{modelo_tipo.replace(' ', '_')}.pkl"
    joblib.dump(modelo, modelo_filename)
    st.success(f"🎯 Modelo de {modelo_tipo} guardado exitosamente en `{modelo_filename}`.")

    # 📌 **Guía Rápida para el Usuario**
    st.info(f"""
    📘 **Guía Rápida**  
    - 🛒 **Clasificación del Tipo de Compra**: Este modelo ayuda a categorizar automáticamente el tipo de compra en base a datos históricos.  
    - ⏳ **Predicción de Tiempos de Entrega**: Estima cuántos días tomará recibir una compra en función del proveedor y la cantidad solicitada.  
    - 🔍 **Cómo Interpretar los Resultados**:  
      - Un menor **MAE** en Predicción de Tiempos de Entrega significa mejores estimaciones.  
      - Una mayor **Precisión** en Clasificación significa que el modelo está clasificando correctamente.  
    """)

    # 📌 **Mostrar parámetros del modelo**
    st.write("📊 **Hiperparámetros del modelo:**", modelo.get_params())
