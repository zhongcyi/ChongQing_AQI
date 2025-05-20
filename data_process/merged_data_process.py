import pandas as pd
import numpy as np

def process_environmental_data(station_file: str,
                               weather_file: str,
                               aqi_file: str,
                               output_file: str,
                               keep_pollutant: str,
                               drop_features: list = None) -> pd.DataFrame:
    """
    合并站点、气象和AQI监测数据，进行时间处理、周期特征生成，并保存为新CSV。

    参数：
    - station_file: 站点及3x3邻域数据 CSV 路径
    - weather_file: 气象数据 CSV 路径
    - aqi_file: AQI监测数据 CSV 路径
    - output_file: 最终保存的合并数据 CSV 路径
    - keep_pollutant: 要保留的污染物字段名，例如 'iPM25IAQI'
    - drop_features: 可选，要在保存前删除的列名列表

    返回：
    - 处理后的DataFrame
    """
    # 读取数据
    stations_df = pd.read_csv(station_file)
    weather_df = pd.read_csv(weather_file)
    aqi_df = pd.read_csv(aqi_file)

    # 合并气象和站点数据
    merged_df = pd.merge(stations_df, weather_df, on='gid', how='left')

    # 删除dt缺失的行
    merged_df.dropna(subset=['dt'], inplace=True)

    # 转换dt列为时间类型
    merged_df['dt'] = pd.to_datetime(merged_df['dt'], errors='coerce')

    # 检查缺失时间间隔（可选打印）
    merged_df['dt_diff'] = merged_df['dt'].diff()
    missing_intervals = merged_df[merged_df['dt_diff'] > pd.Timedelta(hours=1)]
    if not missing_intervals.empty:
        print("缺失的一小时间隔的行:")
        print(missing_intervals[['gid', 'dt', 'dt_diff']])

    # 添加时间周期信息
    merged_df['hour'] = merged_df['dt'].dt.hour
    merged_df['day_of_week'] = merged_df['dt'].dt.dayofweek
    merged_df['month'] = merged_df['dt'].dt.month
    merged_df['quarter'] = merged_df['dt'].dt.quarter

    # 添加正弦周期特征
    merged_df['sin_hour'] = np.sin(2 * np.pi * merged_df['hour'] / 24)
    merged_df['sin_day'] = np.sin(2 * np.pi * merged_df['day_of_week'] / 7)
    merged_df['sin_month'] = np.sin(2 * np.pi * merged_df['month'] / 12)
    merged_df['sin_quarter'] = np.sin(2 * np.pi * merged_df['quarter'] / 4)

    # 合并AQI监测数据
    aqi_df['dReport'] = pd.to_datetime(aqi_df['dReport'])
    merged_df = pd.merge(merged_df, aqi_df, left_on=['dt', 'gid'], right_on=['dReport', 'gid'], how='left')

    # 排序
    df_sorted = merged_df.sort_values(by=['dt', 'id'])

    # 默认删除列
    default_drop = ['dReport', 'hour', 'day_of_week', 'month', 'quarter',
                    'iO3IAQI', 'iCOIAQI', 'iPM25IAQI', 'iNO2IAQI','iPM10IAQI',
                    'istationid', 'iAQI', 'dt_diff']
    default_drop = [x for x in default_drop if x!= keep_pollutant]
    if drop_features:
        default_drop += drop_features
    df_sorted.drop(columns=[col for col in default_drop if col in df_sorted.columns], inplace=True)

    # 编码时间
    df_sorted['time_encoded'] = (df_sorted['dt'] - df_sorted['dt'].min()).dt.total_seconds() // 3600
    df_sorted.drop(columns=['dt'], inplace=True)

    # 列顺序调整
    columns = ['time_encoded', 'gid', 'id'] + [col for col in df_sorted.columns if col not in ['time_encoded', 'gid', 'id']]
    df_reset_col = df_sorted[columns]

    # 保存结果
    df_reset_col.to_csv(output_file+'_'+keep_pollutant+'.csv', index=False)
    print(f"保存成功: {output_file+'_'+keep_pollutant+'.csv'}, 形状: {df_reset_col.shape}")

    return df_reset_col
if __name__ == '__main__':
    "'iO3IAQI', 'iCOIAQI', 'iPM25IAQI', 'iNO2IAQI','iPM10IAQI'"
    '''保存不同的污染物数据集'''
    df = process_environmental_data(
        station_file='target_and_neighbors.csv',
        weather_file='meteorology_data.csv',
        aqi_file='modified_monitoring_data.csv',
        keep_pollutant='iPM10IAQI',
        output_file='merged_data',

    )
