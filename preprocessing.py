import pandas as pd
from pyproj import Transformer

df1 = pd.read_csv('gps.csv', date_parser=['datetime'])
df2 = pd.read_csv('pack.csv', date_parser=['datetime'])

# Загрузка данных
coordinates = pd.read_csv('gps.csv', parse_dates=['datetime'])
orientation = pd.read_csv('pack.csv', parse_dates=['datetime'])
timestamps_df = pd.read_csv('frame_data2.csv', sep=',', header=None, names=['index', 'timestamp'])

# Предобработка временных меток
timestamps_df['timestamp'] = timestamps_df['timestamp'].str.replace(r'[\",]', '', regex=True)
timestamps_df['timestamp'] = pd.to_datetime(timestamps_df['timestamp'].str.strip(), errors='coerce')
timestamps_df.dropna(subset=['timestamp'], inplace=True)

# Округление временных меток
coordinates['datetime'] = coordinates['datetime'].dt.round('ms')
orientation['datetime'] = orientation['datetime'].dt.round('ms')
timestamps_df['timestamp'] = timestamps_df['timestamp'].dt.round('ms')

# Вычисление относительного времени
start_time = min(coordinates['datetime'].min(), orientation['datetime'].min(), timestamps_df['timestamp'].min())
coordinates['time_seconds'] = (coordinates['datetime'] - start_time).dt.total_seconds()
orientation['time_seconds'] = (orientation['datetime'] - start_time).dt.total_seconds()
timestamps_df['time_seconds'] = (timestamps_df['timestamp'] - start_time).dt.total_seconds()

# Объединение данных
merged_data = pd.merge_asof(timestamps_df.sort_values('time_seconds'),
                            coordinates[['time_seconds', 'lat', 'lon']].sort_values('time_seconds'),
                            on='time_seconds', direction='nearest', tolerance=0.019)
merged_data = pd.merge_asof(merged_data.sort_values('time_seconds'),
                            orientation[['time_seconds', 'alt', 'roll', 'pitch', 'yaw']].sort_values('time_seconds'),
                            on='time_seconds', direction='nearest', tolerance=0.019)

# Преобразование географических координат в метрические
transformer = Transformer.from_crs('epsg:4326', 'epsg:32637')  # Используйте соответствующие EPSG коды для вашей области
merged_data['x'], merged_data['y'] = transformer.transform(merged_data['lat'].values, merged_data['lon'].values)
merged_data['z'] = merged_data['alt']

# заполнение пропусков
merged_data[['lat', 'lon', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'alt']] = \
                    merged_data[['lat', 'lon', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'alt']].interpolate(method='spline',
                                                                                                   order=3)

# определение скорости
merged_data['velocity_x'] = merged_data['x'].diff() /  merged_data['time_seconds'].diff()
merged_data['velocity_y'] = merged_data['y'].diff() /  merged_data['time_seconds'].diff()
merged_data['velocity_z'] = merged_data['z'].diff() /  merged_data['time_seconds'].diff()

merged_data['velocity_yaw'] = merged_data['yaw'].diff() /  merged_data['time_seconds'].diff()
merged_data['velocity_pitch'] = merged_data['pitch'].diff() /  merged_data['time_seconds'].diff()
merged_data['velocity_roll'] = merged_data['roll'].diff() /  merged_data['time_seconds'].diff()

# определение ускорения
merged_data['acceleration_x'] = merged_data['velocity_x'].diff() /  merged_data['time_seconds'].diff()
merged_data['acceleration_y'] = merged_data['velocity_y'].diff() /  merged_data['time_seconds'].diff()
merged_data['acceleration_z'] = merged_data['velocity_z'].diff() /  merged_data['time_seconds'].diff()

merged_data['acceleration_yaw'] = merged_data['velocity_yaw'].diff() /  merged_data['time_seconds'].diff()
merged_data['acceleration_pitch'] = merged_data['velocity_pitch'].diff() /  merged_data['time_seconds'].diff()
merged_data['acceleration_roll'] = merged_data['velocity_roll'].diff() /  merged_data['time_seconds'].diff()

merged_data['F'] = merged_data.apply(lambda line: (line['acceleration_x']**2 + line['acceleration_y']**2 + \
                 (line['acceleration_z'] + 9.80665)**2)**.5, axis=1)

merged_data['dt'] = merged_data['time_seconds'].diff()

