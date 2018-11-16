# Create Conda environment
conda create -n ee_py3 -c conda-forge python=3 google-api-python-client pyCrypto spyder jupyter

source activate ee_py3

# Ensure that a crypto library is available
python -c "from oauth2client import crypt"

# Install earthengine-api
pip install earthengine-api

# Authenticate earth engine
earthengine authenticate
# Follow procedure to authenticate and paste the access token in your terminal

# Check if earth engine has been installed
python -c "import ee; ee.Initialize()"
# If you don't get an error, you are good to go
