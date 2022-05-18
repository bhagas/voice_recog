install python 3.8

buat virtual environtment

```python
python -m venv nama_env
```
masuk ke virtualenvirontment
```python
.\nama_env\Scripts\activate
```
install packages

```python
pip install matplotlib
pip install tensorflow
pip install librosa
pip install Ipython
pip install argparse
pip install tabulate
```

download dataset, copy semua actor, buat folder dataset dan taruh di dalam nya

jalankan script

```python
 python .\latihan.py --iterasi 10 --kepadatan 32 --dataset dataset_male --output hasil_prediksi_cowo
```