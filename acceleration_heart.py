import pandas as pd
import numpy as np

# Load your CSV file into a Pandas DataFrame
# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('wash.csv')

# Define attributes for which you want to calculate statistics
attributes = ['ax', 'ay', 'az']  # Add your attribute column names here

# Calculate window size in seconds (6 seconds)
window_size = 6

# Calculate overlap size in seconds (0.6 seconds)
overlap_size = 5.4

# Create a dictionary to store statistics for each attribute
attribute_stats = {attr: {'Mean': [], 'Median': [], 'Std_Dev': [], 'Skewness': [],'MAD':[], 'Kurtosis': []}
                   for attr in attributes}

# Calculate the start and end times based on 'time6' attribute
start_time = df['time6'].min()
end_time = df['time6'].max()

activity_count = []
total_tde = []
spectral_entropy = []
cnt = 0

shr = []
fhr = []
rms_RR_interval = []
mean_RR_interval = []
std_RR_interval = []
sdsd_RR_interval = []
NN50 = []
heart_rate = []

fall_data= []
data = []

while start_time <= (end_time - window_size):
    window_data = df[(df['time6'] >= start_time) & (df['time6'] < start_time + window_size)]
    
    if not window_data.empty:
        data = []
        for attr in attributes:
            attribute_values = window_data[attr]
            attribute_stats[attr]['Mean'].append(np.mean(attribute_values))
            attribute_stats[attr]['Median'].append(np.median(attribute_values))
            attribute_stats[attr]['Std_Dev'].append(np.std(attribute_values))
            attribute_stats[attr]['Skewness'].append(pd.Series(attribute_values).skew())
            attribute_stats[attr]['MAD'].append(np.mean(np.abs(attribute_values - np.mean(attribute_values))))
            attribute_stats[attr]['Kurtosis'].append(pd.Series(attribute_values).kurtosis())

            data.append(np.mean(attribute_values))
            data.append(np.median(attribute_values))
            data.append(np.std(attribute_values))
            data.append(pd.Series(attribute_values).skew())
            data.append(np.mean(np.abs(attribute_values - np.mean(attribute_values))))
            data.append(pd.Series(attribute_values).kurtosis())
            # Combine signals to calculate magnitude
        magnitude = np.sqrt(np.square(window_data['ax']) + np.square(window_data['ay']) + np.square(window_data['az']))
        threshold = 3.0  # Threshold in gravitational units (g)
        # Calculate activity count
        activity_count.append(np.sum(magnitude > threshold))
        data.append(np.sum(magnitude > threshold))

        tde_x = np.sum(np.square(window_data['ax']))  # TDE for x-axis
        tde_y = np.sum(np.square(window_data['ay']))  # TDE for y-axis
        tde_z = np.sum(np.square(window_data['az']))  # TDE for z-axis

        # Total time-domain energy combining all axes
        total_tde.append( tde_x + tde_y + tde_z)
        data.append(tde_x + tde_y + tde_z)

        fft_magnitude = np.fft.fft(magnitude)
        psd = np.abs(fft_magnitude) ** 2
        normalized_psd = psd / np.sum(psd)
            
        spectral_entropy.append(-np.sum(normalized_psd * np.log2(normalized_psd)))
        data.append(-np.sum(normalized_psd * np.log2(normalized_psd)))

        peak_indices = np.where(np.diff(window_data['heart']) == 1023)[0] + 1
        
        if len(peak_indices) >= 2:
            peak_times = window_data['time6'].iloc[peak_indices]
            time_intervals = np.diff(peak_times)
            # heart_rate = 60 / np.mean(time_intervals)  # Calculate heart rate based on mean RR interval
            print("time_intervals : ",time_intervals)
            for i in time_intervals:
                heart_rate.append(60/i)
            shr.append(max(heart_rate))
            fhr.append(min(heart_rate))
            # print(f"Time: {start_time:.2f}-{end_time:.2f}, Heart rate: {heart_rate:.2f} bpm")
            mean_RR_interval.append(np.mean(time_intervals))
            std_RR_interval.append(np.std(time_intervals))
            successive_diff = np.diff(time_intervals)
            std_successive_diff = np.std(successive_diff)
            sdsd_RR_interval.append(std_successive_diff)
            successive_diff2 = [abs(interval) for interval in time_intervals if abs(interval) > threshold]
            NN50.append(len(successive_diff))

            data.append(std_successive_diff)
            data.append(max(heart_rate))
            data.append(min(heart_rate))
            data.append(len(successive_diff))
            

            sum = 0
            cnt =0
            for i in mean_RR_interval:
                sum += i*i
                cnt += 1

            rms_RR_interval.append(np.sqrt(sum/cnt))
            data.append(np.sqrt(sum/cnt))

            fall_data.append(data)
    
    start_time += (window_size - overlap_size)

# Create DataFrames to store the calculated statistics for each attribute
dfs = {attr: pd.DataFrame(stat) for attr, stat in attribute_stats.items()}

# Display the resulting statistics for each attribute
for attr, df in dfs.items():
    print(f"Statistics for '{attr}':")
    # print(df)
    print("\n")

# print("activity_count : ",activity_count)
# print(len(activity_count))
print("                                ")
print("                                ")
# print("total tde : ",total_tde)
# print(len(total_tde))
print("                                ")
print("                                ")
# print("Spectral entropy : ",spectral_entropy)
# print(len(spectral_entropy))

# print("rms of RR intervals : ",rms_RR_interval)
# print("sdsd of RR intervals : ",sdsd_RR_interval)
# print("NN50 : ",NN50)
# print("fast heart rate : ",fhr)
# print("slowest heart rate : ",shr)
print(fall_data)