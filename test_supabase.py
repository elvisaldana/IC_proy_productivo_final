from supabase_api import fetch_data_from_supabase, insert_data_into_supabase

# Prueba de fetch_data_from_supabase
print("Consultando datos de la tabla 'ordencompra':")
data = fetch_data_from_supabase("ordencompra")
print(data)

# Prueba de insert_data_into_supabase
print("Insertando datos de prueba en la tabla 'ordencompra':")
new_data = {
    "codigo_de_compra": "12345",
    "estado": "Pendiente",
    "cantidad": 10,
    
}
response = insert_data_into_supabase("ordencompra", new_data)
print(response)
