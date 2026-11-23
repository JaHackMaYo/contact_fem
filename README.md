# FEM 2D contatto semianello-blocco

## Avvio

```bash
pip install -r requirements.txt
streamlit run app.py
```

Il FEM, la mesh strutturata, l'assemblaggio CST e il contatto penalty sono implementati direttamente nel file `app.py`. NumPy è usato solo per array e algebra lineare; Matplotlib per i grafici; Streamlit per l'interfaccia.

## Unità
Usare un sistema coerente, per esempio N, mm, N/mm². Le reazioni 2D sono per unità di profondità.
