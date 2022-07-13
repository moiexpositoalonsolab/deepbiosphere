from setuptools import setup


setup(
	name='deepbiosphere',
	version='0.0.1',
	author='moiexpositoalonsolab',
# if CUDA is 10.0, you need pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
# rtree also requires libspatialindex-dev so if using pip, must also install  `sudo apt install libspatialindex-dev python-rtree`
# for systems running python 3.5, you should pip install pyproj==2.6.1 and geopandas==0.8.0 to get it to work
	install_requires=["rtree", "geojson", "inplace-abn", "numpy", "tensorboard", "jupyter", "matplotlib", "torch", "pandas", "tqdm" ,  "shapely==1.7.1", 'rasterio', 'geopandas==0.8.1', 'pygeos==0.8', 'scikit-learn']
	)