# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=2m_temperature &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=10m_u_component_of_wind &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=10m_v_component_of_wind &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=geopotential &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=geopotential_500 &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=relative_humidity &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=temperature &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=temperature_850 &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=total_precipitation &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=u_component_of_wind &
# python src/download_weatherbench.py --root=/datadrive/5.625deg --resolution=5.625 --variable=v_component_of_wind

python src/download_weatherbench.py --root=/datadrive/datasets/5.625deg --dataset era5 --resolution=5.625 --variable=specific_humidity &
python src/download_weatherbench.py --root=/datadrive/datasets/5.625deg --dataset era5 --resolution=5.625 --variable=toa_incident_solar_radiation
# python src/download_weatherbench.py --root=/datadrive/datasets/5.625deg --dataset era5 --resolution=5.625 --variable=constants

# python src/download_weatherbench.py --root /datadrive/CMIP6/MPI-ESM/5.625deg --dataset cmip6 --variable geopotential --resolution 5.625 &
# python src/download_weatherbench.py --root /datadrive/CMIP6/MPI-ESM/5.625deg --dataset cmip6 --variable specific_humidity --resolution 5.625 &
# python src/download_weatherbench.py --root /datadrive/CMIP6/MPI-ESM/5.625deg --dataset cmip6 --variable temperature --resolution 5.625 &
# python src/download_weatherbench.py --root /datadrive/CMIP6/MPI-ESM/5.625deg --dataset cmip6 --variable u_component_of_wind --resolution 5.625 &
# python src/download_weatherbench.py --root /datadrive/CMIP6/MPI-ESM/5.625deg --dataset cmip6 --variable v_component_of_wind --resolution 5.625