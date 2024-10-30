import gradio as gr
import pandas as pd
import pickle

def predecir(Nombre, Rareza, Antiguedad, Fullart, Holo, Foil, PSA, Idioma):
    with open('modelo.pkl', 'rb') as file:
        model = pickle.load(file)

    input_data = pd.DataFrame({
        'NOMBRE': [Nombre],
        'RAREZA': [Rareza],
        'ANTIGUEDAD': [Antiguedad],
        'FULLART': [1 if Fullart == "Si" else 0],
        'HOLO': [1 if Holo == "Si" else 0],
        'FOIL': [1 if Foil == "Si" else 0],
        'PSA': [PSA],
        'IDIOMA': [1 if Idioma == "Inglés" else 0]
    })

    model_columns = ['RAREZA', 'ANTIGUEDAD', 'FULLART', 'HOLO', 'FOIL', 'PSA', 'IDIOMA']
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    precio_predicho = model.predict(input_data)

    return precio_predicho[0]


demo = gr.Interface(
    fn=predecir,
    inputs=["text",gr.Slider(value=1, minimum=1, maximum=10, step=1),gr.Slider(value=1996, minimum=1996, maximum=2024, step=1),gr.Radio(["Si", "No"], label="Fullart"),gr.Radio(["Si", "No"], label="Holo"),gr.Radio(["Si", "No"], label="Foil"),gr.Slider(value=1, minimum=1, maximum=10, step=1),gr.Radio(["Inglés", "Japonés"], label="Idioma"),],
    outputs=["text"],
    allow_flagging='never'
)
demo.launch(share=True)

