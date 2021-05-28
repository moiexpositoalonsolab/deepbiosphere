from setuptools import setup


setup(
	name='deepbiosphere',
	version='0.0.1',
	author='moiexpositoalonsolab',
# if CUDA is 10.0, you need pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
# rtree also requires libspatialindex-dev so if using pip, must also install  `sudo apt install libspatialindex-dev python-rtree`
	install_requires=["rtree", "sklearn", "geojson", "inplace-abn", "numpy", "tensorboard", "jupyter", "matplotlib", "torch", "pandas", "tqdm" , "deepdish", "shapely", 'rasterio']
	)
