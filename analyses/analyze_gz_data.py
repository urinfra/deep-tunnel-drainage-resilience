"""
分析广州东濠涌深隧真实运行数据 (2022-2025)
提取关键运行参数用于SWMM模型率定
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

INPUT_DIR = r"C:\ShareCache\林沛元\科研与学习\Paper\Journal Paper\TUNNEL ENGINEERING\Paper 5 Safety's Shadow\Inputs"
OUTPUT_DIR = r"C:\ShareCache\林沛元\科研与学习\Paper\Journal Paper\TUNNEL ENGINEERING\Paper 5 Safety's Shadow\Outputs"

FILES = ['深隧数据2022.xlsx', '深隧数据2023.xlsx', '深隧数据2024.xlsx', '深隧数据2025.xlsx']

def excel_serial_to_date(serial):
    """Convert Excel serial date number to datetime"""
    if pd.isna(serial):
        return None
    try:
        return datetime(1899, 12, 30) + timedelta(days=int(serial))
    except:
        return None

def parse_rainfall_data(filepath, year):
    """Parse rainfall sheet: single row with daily rainfall columns"""
    try:
        xl = pd.ExcelFile(filepath)
        rain_sheet = [s for s in xl.sheet_names if 'A-' in s and '降雨' in s and '经纬度' not in s][0]
        df = pd.read_excel(filepath, sheet_name=rain_sheet, header=None)
    except:
        return None

    # Find the data row (after header rows)
    data = {}
    for i in range(df.shape[0]):
        row = df.iloc[i]
        # Find columns that are Excel date serials (43000-46000 range for 2017-2026)
        rainfall_values = {}
        for j in range(2, len(row)):
            col_name = df.iloc[0, j] if not pd.isna(df.iloc[0, j]) else j
            val = row.iloc[j]
            if isinstance(col_name, (int, float)) and not pd.isna(col_name) and 43000 <= col_name <= 47000:
                date = excel_serial_to_date(col_name)
                if date and not pd.isna(val):
                    rainfall_values[date] = float(val)

        if rainfall_values:
            data['rainfall'] = rainfall_values
            break

    return data

def parse_peak_storage(filepath):
    """Parse peak storage sheet"""
    xl = pd.ExcelFile(filepath)
    sheet = [s for s in xl.sheet_names if '峰值蓄水量' in s][0]
    df = pd.read_excel(filepath, sheet_name=sheet)
    df.columns = ['date_serial', 'peak_storage_m3']

    records = []
    for _, row in df.iterrows():
        serial = row['date_serial']
        val = row['peak_storage_m3']
        if pd.isna(serial) or pd.isna(val):
            continue
        try:
            date = excel_serial_to_date(float(serial))
            storage = float(val)
            if date and storage > 0:
                records.append({'date': date, 'peak_storage_m3': storage})
        except:
            continue
    return records

def parse_pump_discharge(filepath):
    """Parse pump station discharge data"""
    xl = pd.ExcelFile(filepath)
    sheet = [s for s in xl.sheet_names if '瞬时排水流量' in s][0]
    df = pd.read_excel(filepath, sheet_name=sheet)
    total_sheet = [s for s in xl.sheet_names if '泵站总排水量' in s][0]
    df_total = pd.read_excel(filepath, sheet_name=total_sheet)

    result = {}
    if df.shape[0] > 0:
        result['min_flow_m3h'] = float(df.iloc[0, 1]) if df.shape[1] > 1 else None
        result['max_flow_m3h'] = float(df.iloc[0, 2]) if df.shape[1] > 2 else None
    if df_total.shape[0] > 0:
        result['annual_total_m3'] = float(df_total.iloc[0, 1]) if df_total.shape[1] > 1 else None
    return result

def parse_emptying_time(filepath):
    """Parse tunnel emptying time records"""
    xl = pd.ExcelFile(filepath)
    sheet = [s for s in xl.sheet_names if '排空时间' in s][0]
    df = pd.read_excel(filepath, sheet_name=sheet)

    records = []
    current_date = None
    for _, row in df.iterrows():
        serial = row.iloc[0]
        time_val = row.iloc[1]
        water_level = row.iloc[2] if df.shape[1] > 2 else None

        if not pd.isna(serial):
            if isinstance(serial, datetime):
                current_date = serial
            else:
                try:
                    d = excel_serial_to_date(float(serial))
                    if d:
                        current_date = d
                except (ValueError, TypeError):
                    pass

        if current_date and not pd.isna(time_val):
            time_str = str(time_val).strip()
            records.append({
                'date': current_date,
                'time': time_str,
                'water_level': str(water_level) if not pd.isna(water_level) else None
            })
    return records

def parse_maintenance(filepath):
    """Parse maintenance records"""
    xl = pd.ExcelFile(filepath)
    result = {}

    # Sediment records
    sed_sheet = [s for s in xl.sheet_names if '沉积物厚度' in s]
    if sed_sheet:
        df = pd.read_excel(filepath, sheet_name=sed_sheet[0])
        result['sediment_cm'] = str(df.iloc[0, -1]) if df.shape[0] > 0 else None

    # Desilting records
    desilt_sheet = [s for s in xl.sheet_names if '清淤' in s]
    if desilt_sheet:
        df = pd.read_excel(filepath, sheet_name=desilt_sheet[0])
        total_sediment_removed = 0
        for _, row in df.iterrows():
            for col in df.columns:
                if '运输量' in str(col) or 't' in str(col):
                    val = row[col]
                    if not pd.isna(val):
                        try:
                            total_sediment_removed += float(val)
                        except:
                            pass
        result['total_sediment_removed_t'] = total_sediment_removed
        result['desilting_operations'] = df.shape[0]

    # Equipment failures
    fault_sheet = [s for s in xl.sheet_names if '故障' in s or '维修' in s]
    if fault_sheet:
        df = pd.read_excel(filepath, sheet_name=fault_sheet[0])
        faults = []
        for _, row in df.iterrows():
            faults.append({
                'equipment': str(row.iloc[1]) if df.shape[1] > 1 else '',
                'date': row.iloc[2] if df.shape[1] > 2 else '',
                'type': str(row.iloc[3]) if df.shape[1] > 3 else '',
                'issue': str(row.iloc[-1]) if df.shape[1] > 4 else ''
            })
        result['faults'] = faults
        result['num_faults'] = len(faults)

    return result

# ============ MAIN ANALYSIS ============
print("=" * 80)
print("广州东濠涌深隧运行数据分析报告")
print("=" * 80)

all_peak_storage = []
all_pump_data = {}
all_maintenance = {}
all_rainfall = {}

for f in FILES:
    filepath = os.path.join(INPUT_DIR, f)
    year = int(f.replace('深隧数据', '').replace('.xlsx', ''))
    print(f"\n{'='*60}")
    print(f"  {year} 年")
    print(f"{'='*60}")

    # 1. Peak storage
    storage_records = parse_peak_storage(filepath)
    if storage_records:
        storages = [r['peak_storage_m3'] for r in storage_records]
        all_peak_storage.extend(storage_records)
        print(f"\n[蓄水量] {len(storage_records)} 次事件记录")
        print(f"  峰值蓄水量: 最小={min(storages):.0f} m3, 最大={max(storages):.0f} m3, "
              f"均值={np.mean(storages):.0f} m3, 中位数={np.median(storages):.0f} m3")
        print(f"  超过设计标准50%的事件: {sum(1 for s in storages if s > 50000)} 次")
        print(f"  超过设计标准80%的事件: {sum(1 for s in storages if s > 80000)} 次")

    # 2. Pump discharge
    pump = parse_pump_discharge(filepath)
    if pump:
        all_pump_data[year] = pump
        print(f"\n[泵站] 瞬时流量范围: {pump.get('min_flow_m3h')} - {pump.get('max_flow_m3h')} m3/h")
        print(f"  年总排水量: {pump.get('annual_total_m3', 0):,.0f} m3")

    # 3. Emptying time
    emptying = parse_emptying_time(filepath)
    if emptying:
        durations = []
        for r in emptying:
            time_str = r['time']
            if ':' in str(time_str) and '24:00' not in str(time_str):
                try:
                    parts = str(time_str).split(':')
                    hours = float(parts[0]) + float(parts[1])/60
                    if hours > 0 and hours < 48:
                        durations.append(hours)
                except:
                    pass
        if durations:
            print(f"\n[排空] {len(emptying)} 条记录, {len(durations)} 个有效排空时长")
            print(f"  排空时长: 最小={min(durations):.1f}h, 最大={max(durations):.1f}h, "
                  f"均值={np.mean(durations):.1f}h, 中位数={np.median(durations):.1f}h")

    # 4. Maintenance
    maint = parse_maintenance(filepath)
    if maint:
        all_maintenance[year] = maint
        print(f"\n[运维] 沉积物厚度: {maint.get('sediment_cm', 'N/A')}")
        print(f"  清淤作业: {maint.get('desilting_operations', 0)} 次, "
              f"总清淤量: {maint.get('total_sediment_removed_t', 0):.1f} t")
        print(f"  设备故障/维修: {maint.get('num_faults', 0)} 次")
        if 'faults' in maint:
            for fault in maint['faults']:
                print(f"    - [{fault['type']}] {fault['equipment']}: {fault['issue'][:80]}")

# ============ CROSS-YEAR SUMMARY ============
print(f"\n\n{'='*80}")
print("  跨年度汇总统计")
print(f"{'='*80}")

if all_peak_storage:
    storages = [r['peak_storage_m3'] for r in all_peak_storage]
    dates = [r['date'] for r in all_peak_storage]
    print(f"\n总运行事件: {len(all_peak_storage)} 次 (2022-2025)")
    print(f"蓄水量分布:")
    print(f"  P50: {np.percentile(storages, 50):.0f} m3")
    print(f"  P90: {np.percentile(storages, 90):.0f} m3")
    print(f"  P95: {np.percentile(storages, 95):.0f} m3")
    print(f"  P99: {np.percentile(storages, 99):.0f} m3")
    print(f"  Max: {max(storages):.0f} m3")

    # Estimate return periods from observed data
    n_years = 4
    n_events = len(storage_records) if storage_records else 1
    sorted_storages = sorted(storages, reverse=True)
    print(f"\n基于4年观测的经验重现期估计:")
    for rank, s in enumerate(sorted_storages[:10], 1):
        rp = (n_years + 0.12) / (rank - 0.44)  # Gringorten formula
        print(f"  Rank {rank}: {s:.0f} m3 -> 经验重现期 ~{rp:.1f} 年")

print(f"\n\n泵站年度排水量:")
for year in sorted(all_pump_data.keys()):
    p = all_pump_data[year]
    print(f"  {year}: {p.get('annual_total_m3', 0):,.0f} m3, "
          f"流量范围 {p.get('min_flow_m3h')}-{p.get('max_flow_m3h')} m3/h")

print(f"\n沉积物累积趋势:")
for year in sorted(all_maintenance.keys()):
    m = all_maintenance[year]
    print(f"  {year}: 厚度={m.get('sediment_cm', 'N/A')}, "
          f"清淤={m.get('desilting_operations', 0)}次, "
          f"清淤量={m.get('total_sediment_removed_t', 0):.1f}t, "
          f"故障={m.get('num_faults', 0)}次")

print(f"\n\n分析完成。数据已可用于SWMM模型率定。")
