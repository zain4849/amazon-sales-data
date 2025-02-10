import streamlit as st
import streamlit_themes as st_themes

st_themes.set_preset_theme('Ocean')

Visualization = st.Page(r'pages/Visualization.py', title='Visualization')
Main_Page = st.Page(r'pages/Main_Page.py', title='Main_Page')
Model=st.Page(r'pages/Model.py', title='Model')

pg=st.navigation(
    {
        'Main_Page': [Main_Page], 
        'Visualization': [Visualization],
        'Model': [Model]
    }
)


pg.run()