import streamlit as st
import mysql.connector
import os

st.title("TiDB SSL Test")

ca_content = st.secrets["TIDB_SSL_CA"]
ca_path = "/tmp/test_tidb_ca.pem"
if not os.path.exists(ca_path):
    with open(ca_path, "w") as f:
        f.write(ca_content)

db = st.secrets["connections"]["databases"]["default"]

try:
    conn = mysql.connector.connect(
        host=db["host"],
        port=int(db["port"]),
        user=db["username"],
        password=db["password"],
        database=db["database"],
        ssl_ca=ca_path,
        ssl_verify_cert=True
    )
    st.success("Connected successfully!")
    conn.close()
except Exception as e:
    st.error(f"Connection failed: {str(e)}")
